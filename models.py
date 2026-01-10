from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class Prompt(str, Enum):
    NORMAL = "normal"
    WEB_SEARCH = "web_search"
    # CHOOSE_FOR_ME = "choose_for_me"
    # GENERATE_FEEDBACK = "generate_feedback"
    # DECISION_CHAT = "decision_chat"

    def to_json(self):
        return self.value

    def __json__(self):
        return self.value


class PromptSelection(BaseModel):
    prompt: str = Field(
        description="The prompt to use to answer the user's question most effectively. Return the prompt title and nothing else."
    )


class RoleplayScenarioResponse(BaseModel):
    response: str = Field(description="The response of the narrator.")
    suggested_choices: List[str] = Field(
        description="A list of 3 choices that the user could make to move forward in the scenario. Keep these choices to less than 10 words long. Be as specific as possible.",
    )


class MessageResponse(BaseModel):
    text: str = Field(description="Your response to the conversation.")
    suggested_messages: List[str] = Field(
        ...,
        description="A list of suggested messages that the user could send back to the response you generate. The messages should be from the user's point of view in the conversation. If you are asking me a question in your response, the suggested messages should be answers to that question. The suggested messages should be 5-15 words long.",
    )


class RouterResponse(BaseModel):
    response: BaseModel = Field(description="The response of the router.")
    prompt: Prompt = Field(
        description="The prompt that was used to generate the response."
    )


class UIResponse(BaseModel):
    text: str = Field(description="The text to display to the user.")
    citations: List[str] = Field(
        default=[],
        description="A list of citations for the response. These should be links to the sources of the information in the response.",
    )
    suggested_messages: List[str] = Field(
        default=[],
        description="A list of suggested follow-up messages that the user might want to send.",
    )


class Choice(BaseModel):
    name: str = Field(
        ...,
        description="The name of the choice that this outcome is for. This should be a single word or phrase. It should be an action that I can take.",
    )
    assumptions: List[str] = Field(
        ...,
        description="A list of 1 or 2 assumptions that were made to generate this outcome. These assumptions must materially affect the outcome of the choice. If the assumption does not affect whether I make this choice, then do not include it. Use at most 8 words for each assumption.",
    )

    best_case_scenario: str = Field(
        ..., description="The best case scenario for this outcome. Use at most 8 words."
    )
    worst_case_scenario: str = Field(
        ...,
        description="The worst case scenario for this outcome. Use at most 8 words.",
    )

    def __str__(self):
        return (
            f"## {self.name}\n"
            f"### Scenarios:\n"
            f"- Best: {self.best_case_scenario}\n"
            f"- Worst: {self.worst_case_scenario}\n\n"
        )


class ChoicesResponse(BaseModel):
    choices: List[Choice] = Field(description="A list of choices.")
    title: str = Field(
        description="A question that summarizes the decision being considered. The choices should be answers to this question. An example: 'Should I stay in my current job?'"
    )
    uncertainties: List[str] = Field(
        ...,
        description="A list of 1 or 2 uncertainties about the decision being considered. These uncertainties must materially affect the quality of the choice. If the uncertainty does not affect whether I make this choice, then do not include it. A good uncertainty captures what would need to be true for this to be the right decision or wrong decision, depending on the answer. Each uncertainty should be a question that ends with a question mark. Use at most 8 words for each uncertainty.",
    )
    next_steps: List[str] = Field(
        ...,
        description="A list of 1 or 2 next steps for this decision. These next steps should be actions that I can take or information that I can gather to reduce my uncertainty about the decision. Use at most 8 words for each next step.",
    )


class ChooseForMeResponse(BaseModel):
    explanation: str = Field(
        description="A short explanation of why this choice could be a good decision. Use at most 30 words. Write as though you are speaking to me, and use words that I would expect to hear from a friend as a suggestion. The explanation should be a single complete sentence."
    )
    chosen_index: int = Field(
        description="The index of the chosen option. The index starts at 0. It can go up to the number of options minus 1."
    )


class Character(BaseModel):
    name: str
    age: int
    gender: str
    location: str
    occupation: str
    mood: str
    facts: List[str] = Field(..., description="A list of 3 facts about the character")
    goals: List[str] = Field(
        ..., description="A list of 3 goals the character is trying to achieve"
    )
    fears: List[str] = Field(
        ..., description="A list of 3 fears the character is experiencing"
    )
    desires: List[str] = Field(
        ..., description="A list of 3 desires the character is experiencing"
    )
    joys: List[str] = Field(
        ..., description="A list of 3 joys the character is experiencing"
    )
    problems: List[str] = Field(
        ..., description="A list of 3 problems the character is facing"
    )


class Message(BaseModel):
    role: str
    content: str


class MessageList(BaseModel):
    messages: List[Message] = Field(description="A list of messages.")

    def __str__(self):
        return "\n".join(
            [f"{message.role}: {message.content}" for message in self.messages]
        )

    def add_message(self, message: Message):
        self.messages.append(message)

    def prepend_message(self, message: Message):
        self.messages.insert(0, message)

    def clear(self):
        self.messages = []

    def return_openai_format(self):
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]

    def return_persona_format(self):
        message_list = []
        for message in self.messages:
            if message.role == "user":
                message_list.append(f"user: {message.content}")
            else:
                message_list.append(f"persona: {message.content}")
        return "\n".join(message_list)

    def return_json(self):
        return [message.model_dump() for message in self.messages]


class NextStepsResponse(BaseModel):
    next_steps: List[str] = Field(
        ...,
        min_items=1,  # Ensures at least 1 step
        max_items=2,  # Limits to no more than 2 steps
        description="A list of 1-2 specific, actionable next steps for the user. Each step should be clear, concrete, and start with a verb. Each step should be 5-15 words and directly applicable to the situation.",
    )


class AdditionalActionResponse(BaseModel):
    additional_action: str = Field(
        description="A single additional action suggestion that complements the user's existing actions. The action should be specific, actionable, start with a verb, and be 5-15 words long."
    )
