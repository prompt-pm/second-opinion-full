"""Widget tools that the AI can call during conversation."""

# OpenAI function calling format
WIDGET_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "show_story_prompts",
            "description": "Show best-case/worst-case story prompts to help user discover what matters to them. Use when the user can't articulate their priorities, says things like 'I don't know what I want', or seems confused about what matters.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_card_sort",
            "description": "Show common priorities for this decision type as selectable cards. Use when the user is slow to identify priorities, lists too many vague factors, or would benefit from seeing common options.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision_type": {
                        "type": "string",
                        "enum": ["job", "housing", "relationship", "purchase", "other"],
                        "description": "The type of decision to show relevant priority cards for",
                    }
                },
                "required": ["decision_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_priority_tournament",
            "description": "Show pairwise comparisons to help rank priorities. Use when the user says 'everything is equally important', can't decide on ranking, or is paralyzed trying to order their priorities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "priorities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of priorities to rank through pairwise comparison",
                    }
                },
                "required": ["priorities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_budget_allocation",
            "description": "Show sliders to allocate 100 points across priorities. Use to confirm a ranking, when the user wants to express precise weights, or after a tournament to fine-tune.",
            "parameters": {
                "type": "object",
                "properties": {
                    "priorities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of priorities to allocate points to",
                    }
                },
                "required": ["priorities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_elimination_game",
            "description": "Show priorities and ask user to eliminate the least important one at a time. Use when the user has too many priorities (5+) and is overwhelmed, or needs help narrowing down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "priorities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of priorities to eliminate from",
                    }
                },
                "required": ["priorities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_recommendation",
            "description": "Show a scored recommendation comparing options against the user's priorities. Use when you have enough information about both the user's priorities and their options to make a recommendation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "scores": {
                                    "type": "object",
                                    "description": "Scores per priority",
                                },
                            },
                        },
                        "description": "The options to compare with their scores",
                    },
                    "priorities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "weight": {"type": "number"},
                            },
                        },
                        "description": "The priorities with their weights (should sum to 1.0)",
                    },
                },
                "required": ["options", "priorities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_tradeoff_acknowledgment",
            "description": "Show what the user gains vs sacrifices with their choice. Use before finalizing a decision to ensure the user explicitly accepts the tradeoffs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "description": "The option the user is choosing",
                    },
                    "gaining": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What the user gains by making this choice",
                    },
                    "sacrificing": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What the user gives up by making this choice",
                    },
                },
                "required": ["choice", "gaining", "sacrificing"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_premortem",
            "description": "Show the worst realistic outcome and ask if the user can live with it. Use when the user is anxious about committing, keeps second-guessing, or needs to confront the downside.",
            "parameters": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "description": "The option the user is considering",
                    },
                    "worst_case": {
                        "type": "string",
                        "description": "A realistic worst-case scenario if they make this choice",
                    },
                },
                "required": ["choice", "worst_case"],
            },
        },
    },
]

# Card sets for different decision types
CARD_SETS = {
    "job": [
        {"emoji": "💰", "label": "Compensation", "desc": "Salary, bonus, equity"},
        {"emoji": "🏠", "label": "Flexibility", "desc": "Remote work, hours, location"},
        {"emoji": "📈", "label": "Growth", "desc": "Learning, promotion path"},
        {"emoji": "👥", "label": "Culture", "desc": "Team, values, management"},
        {"emoji": "⚖️", "label": "Work-life balance", "desc": "Hours, stress, boundaries"},
        {"emoji": "🎯", "label": "Mission", "desc": "Meaningful work, impact"},
        {"emoji": "🔒", "label": "Stability", "desc": "Job security, company runway"},
        {"emoji": "🚀", "label": "Challenge", "desc": "Hard problems, stretch assignments"},
        {"emoji": "🏆", "label": "Prestige", "desc": "Brand, resume value"},
    ],
    "housing": [
        {"emoji": "💰", "label": "Price", "desc": "Rent/mortgage, total cost"},
        {"emoji": "📍", "label": "Location", "desc": "Neighborhood, area"},
        {"emoji": "🚗", "label": "Commute", "desc": "Time to work, transportation"},
        {"emoji": "📐", "label": "Space", "desc": "Square footage, rooms"},
        {"emoji": "🔒", "label": "Safety", "desc": "Crime rate, security"},
        {"emoji": "☀️", "label": "Natural light", "desc": "Windows, sun exposure"},
        {"emoji": "🏋️", "label": "Amenities", "desc": "Gym, pool, laundry"},
        {"emoji": "🔇", "label": "Quiet", "desc": "Noise level, neighbors"},
        {"emoji": "🐕", "label": "Pet-friendly", "desc": "Pet policies, nearby parks"},
    ],
    "relationship": [
        {"emoji": "💕", "label": "Compatibility", "desc": "Shared interests, lifestyle"},
        {"emoji": "🎯", "label": "Values", "desc": "Beliefs, life priorities"},
        {"emoji": "✨", "label": "Attraction", "desc": "Physical, emotional connection"},
        {"emoji": "🔮", "label": "Future goals", "desc": "Marriage, kids, career"},
        {"emoji": "💬", "label": "Communication", "desc": "Openness, conflict resolution"},
        {"emoji": "🤝", "label": "Trust", "desc": "Reliability, honesty"},
        {"emoji": "👨‍👩‍👧", "label": "Family approval", "desc": "How families get along"},
        {"emoji": "🌍", "label": "Location", "desc": "Where to live, long distance"},
    ],
    "purchase": [
        {"emoji": "💰", "label": "Price", "desc": "Cost, value for money"},
        {"emoji": "⭐", "label": "Quality", "desc": "Build, durability"},
        {"emoji": "🎨", "label": "Aesthetics", "desc": "Look, design, style"},
        {"emoji": "🔧", "label": "Features", "desc": "Functionality, specs"},
        {"emoji": "🏷️", "label": "Brand", "desc": "Reputation, status"},
        {"emoji": "📦", "label": "Convenience", "desc": "Availability, delivery"},
        {"emoji": "🌱", "label": "Sustainability", "desc": "Environmental impact"},
        {"emoji": "🔄", "label": "Resale value", "desc": "Future value, trade-in"},
    ],
    "other": [
        {"emoji": "💰", "label": "Cost", "desc": "Financial impact"},
        {"emoji": "⏰", "label": "Time", "desc": "Time investment required"},
        {"emoji": "😊", "label": "Happiness", "desc": "Personal fulfillment"},
        {"emoji": "👥", "label": "Relationships", "desc": "Impact on others"},
        {"emoji": "🎯", "label": "Goals", "desc": "Alignment with objectives"},
        {"emoji": "⚡", "label": "Energy", "desc": "Effort and motivation"},
        {"emoji": "🔒", "label": "Security", "desc": "Risk and safety"},
        {"emoji": "🌱", "label": "Growth", "desc": "Learning and development"},
    ],
}
