import instructor
import os
from openai import OpenAI

from cerebras.cloud.sdk import Cerebras
from groq import Groq
from models import (
    ChoicesResponse,
    ChooseForMeResponse,
    MessageResponse,
    Prompt,
    PromptSelection,
    Choice,
    RouterResponse,
    UIResponse,
    NextStepsResponse,
    AdditionalActionResponse,
)

from prompts import (
    CHOOSE_OPTION_PROMPT,
    GENERATE_ALTERNATIVE_PROMPT,
    GENERATE_FEEDBACK_PROMPT,
    GENERATE_OUTCOMES_PROMPT,
    GENERATE_QUESTIONS_PROMPT,
    NORMAL_PROMPT,
    PROMPT_SELECTION_PROMPT,
    WEB_SEARCH_PROMPT,
    PRE_DECISION_NEXT_STEPS_PROMPT,
    POST_DECISION_NEXT_STEPS_PROMPT,
    SUGGEST_ADDITIONAL_ACTION_PROMPT,
)
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

CEREBRAS_MODEL = "llama-3.3-70b"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PPLX_API_KEY = os.environ.get("PPLX_API_KEY")


def groq_or_cerebras(messages, temperature, response_model=None):
    """
    Try Groq first, if that fails, try Cerebras.
    """
    try:
        if response_model is not None:
            client = instructor.from_groq(Groq(api_key=GROQ_API_KEY))
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                response_model=response_model,
                max_retries=2,
            )
        else:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_retries=2,
            )
    except Exception as e:
        print(f"Error with Groq: {e}")
        if response_model is not None:
            client = instructor.from_cerebras(
                Cerebras(), mode=instructor.Mode.CEREBRAS_JSON
            )
            response = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                response_model=response_model,
                max_retries=2,
            )
        else:
            client = Cerebras()
            response = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_retries=2,
            )
    return response


def cerebras_router(message_history):
    """
    Route to the appropriate ai prompt based on the input messages.

    Returns:
    {
        "prompt": Prompt,
        "response": Object or JSON dictionary
    }

    The response object will either be an object like ChoicesResponse, Choice, or a JSON dictionary with the following keys:
    {
        "text": str,
        "citations": list[str]
    }
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_router") as span:
        cleaned_messages = clean_up_messages(message_history)
        messages = [
            {
                "role": "user",
                "content": PROMPT_SELECTION_PROMPT.format(
                    message_history=str(cleaned_messages)
                ),
            }
        ]
        router_response = groq_or_cerebras(messages, 0.2, PromptSelection)
        prompt = router_response.prompt
        last_message = cleaned_messages[-1]["content"]
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.INPUT_VALUE, str(last_message))
        # Map prompts to their handler functions
        handlers = {
            Prompt.WEB_SEARCH: lambda: perplexity_web_search(cleaned_messages),
            Prompt.NORMAL: lambda: cerebras_normal(cleaned_messages),
        }
        # Check if we have a handler for this prompt type
        if prompt not in handlers:
            return cerebras_normal(
                cleaned_messages
            )  # Default to normal chat if no specific handler

        # Execute the handler and return results
        full_response = handlers[prompt]()
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(full_response))
        return RouterResponse(prompt=prompt, response=full_response)


def perplexity_web_search(message_history):
    """
    Perform a web search for a given situation using Perplexity.
    """
    pplx_client = OpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")

    # Extract the last message
    last_message = message_history[-1]
    # Get all previous messages for context
    previous_messages = message_history[:-1]

    # Add the final message with the web search prompt
    previous_messages.append(
        {
            "role": "user",
            "content": WEB_SEARCH_PROMPT.format(
                conversation=last_message.get("content", "")
            ),
        }
    )

    response = pplx_client.chat.completions.create(
        model="sonar",
        messages=previous_messages,
    )
    if hasattr(response, "citations"):
        return UIResponse(
            text=response.choices[0].message.content, citations=response.citations
        )
    else:
        return UIResponse(text=response.choices[0].message.content)


def cerebras_normal(message_history):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_normal") as span:
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, NORMAL_PROMPT)
        span.set_attribute(SpanAttributes.INPUT_VALUE, str(message_history))
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        messages = [{"role": "system", "content": NORMAL_PROMPT}] + message_history
        response = groq_or_cerebras(messages, 1.0, MessageResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return UIResponse(
        text=response.text, suggested_messages=response.suggested_messages
    )


def cerebras_generate_outcomes(message_history):
    """
    Generate 3 outcomes for a given situation.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_generate_outcomes") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, GENERATE_OUTCOMES_PROMPT)
        messages = [
            {"role": "system", "content": GENERATE_OUTCOMES_PROMPT}
        ] + message_history
        response = groq_or_cerebras(messages, 1.0, ChoicesResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def cerebras_generate_alternative(situation, results):
    """
    Generate a new, unique alternative that wasn't previously considered. The alternative should be realistic and relevant to the situation.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_generate_alternative") as span:
        # Extract situation context from results if situation is empty
        if not situation and results and "title" in results:
            situation = results.get("title", "")

        prompt = GENERATE_ALTERNATIVE_PROMPT.format(
            situation=situation, results=results
        )
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE, GENERATE_ALTERNATIVE_PROMPT
        )
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str({"situation": situation, "results": results}),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, situation)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 1.0, Choice)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def cerebras_choose_option(situation, results, current_selected_index=None):
    """
    Choose an option based on the situation and results.
    If current_selected_index is provided, choose a different option.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_choose_option") as span:
        # Extract situation context from results if situation is empty
        if not situation and results and "title" in results:
            situation = results.get("title", "")

        total_options = len(results.get("choices", []))
        # Update prompt to include current selection if it exists
        if current_selected_index is not None:
            prompt = CHOOSE_OPTION_PROMPT.format(
                situation=situation,
                results=results,
                current_choice=current_selected_index,
                total_options=total_options,
            )
        else:
            prompt = CHOOSE_OPTION_PROMPT.format(
                situation=situation,
                results=results,
                current_choice="None",
                total_options=total_options,
            )
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, CHOOSE_OPTION_PROMPT)
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str(
                {
                    "situation": situation,
                    "results": results,
                    "current_selected_index": current_selected_index,
                }
            ),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, situation)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 1.0, ChooseForMeResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def generate_question(situation, results, questions=None):
    """
    Generate a question for a given situation and results.
    """
    tracer = trace.get_tracer(__name__)
    if not situation and results and "title" in results:
        situation = results.get("title", "")

    with tracer.start_as_current_span("groq_generate_questions") as span:
        prompt = GENERATE_QUESTIONS_PROMPT.format(
            situation=situation, results=results, questions=questions
        )
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE, GENERATE_QUESTIONS_PROMPT
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 0.9)
        response_message = response.choices[0].message.content
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response_message))
    return response_message


def clean_up_messages(messages):
    # Keep track of total characters
    total_chars = sum(len(msg["content"]) for msg in messages)

    # Target is ~30K chars (roughly 8K tokens)
    while total_chars > 30000:
        if len(messages) <= 2:
            # If only system prompt and last message remain, truncate last message
            last_msg = messages[-1]["content"]
            # Keep first 1000 chars and last 1000 chars to maintain context
            if len(last_msg) > 2000:
                messages[-1]["content"] = last_msg[:1000] + "..." + last_msg[-1000:]
        else:
            # Remove oldest non-system messages in pairs
            for i in range(1, len(messages) - 1):
                if messages[i]["role"] == "user":
                    # Remove this user message and next assistant message if it exists
                    messages.pop(i)
                    if i < len(messages) and messages[i]["role"] == "assistant":
                        messages.pop(i)
                    break

        total_chars = sum(len(msg["content"]) for msg in messages)

    # Ensure messages only have role and content attributes that are STRINGS
    cleaned_messages = []
    for msg in messages:
        cleaned_messages.append({"role": msg["role"], "content": str(msg["content"])})
    messages = cleaned_messages
    return messages


def cerebras_generate_next_steps(
    situation, choice_name=None, results=None, choice_index=None
):
    """
    Generate actionable next steps for a situation, either before or after a decision.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_generate_next_steps") as span:
        # Determine if it's post-decision (choice provided) or pre-decision (no choice)
        if choice_name or choice_index is not None:
            # Post-decision scenario
            if choice_index is not None and results and "choices" in results:
                try:
                    selected_choice = results["choices"][choice_index]
                except (IndexError, TypeError):
                    selected_choice = {"name": choice_name} if choice_name else {}
            else:
                selected_choice = {"name": choice_name} if choice_name else {}

            prompt = POST_DECISION_NEXT_STEPS_PROMPT.format(
                situation=situation,
                results=results if results else {},
                choice_name=choice_name if choice_name else "",
                selected_choice=selected_choice,
            )
            prompt_template = POST_DECISION_NEXT_STEPS_PROMPT
        else:
            # Pre-decision scenario
            prompt = PRE_DECISION_NEXT_STEPS_PROMPT.format(
                situation=situation, results=results if results else {}
            )
            prompt_template = PRE_DECISION_NEXT_STEPS_PROMPT

        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, prompt_template)
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str(
                {
                    "situation": situation,
                    "choice_name": choice_name,
                    "results": results,
                    "choice_index": choice_index,
                }
            ),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 1.0, NextStepsResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def cerebras_suggest_additional_action(situation, existing_next_steps, results=None):
    """
    Generate a single additional action that complements existing actions.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_suggest_additional_action") as span:
        prompt = SUGGEST_ADDITIONAL_ACTION_PROMPT.format(
            situation=situation,
            existing_next_steps=existing_next_steps,
            results=results if results else {},
        )

        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE, SUGGEST_ADDITIONAL_ACTION_PROMPT
        )
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str(
                {
                    "situation": situation,
                    "existing_next_steps": existing_next_steps,
                    "results": results,
                }
            ),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 1.0, AdditionalActionResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response
