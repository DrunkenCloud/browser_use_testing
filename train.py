import json
import asyncio
from dotenv import load_dotenv
from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm import ChatGoogle

load_dotenv()

teacher_llm = ChatGoogle(model="gemini-2.0-flash-exp")

browser_profile = BrowserProfile(
    headless=True,
    viewport={"width": 1280, "height": 1100},
    user_data_dir=None,
)

browser_session = BrowserSession(browser_profile=browser_profile)

THOUGHTS_LOG_PATH = "thoughts_log.jsonl"

async def my_step_hook(agent):
    if not hasattr(my_step_hook, "last_logged_index"):
        my_step_hook.last_logged_index = 0

    model_thoughts = agent.state.history.model_thoughts()
    new_thoughts = model_thoughts[my_step_hook.last_logged_index:]

    if not new_thoughts:
        return

    with open(THOUGHTS_LOG_PATH, "a", encoding="utf-8") as f:
        for thought in new_thoughts:
            log_entry = {
                "thinking": thought.thinking,
                "evaluation_previous_goal": thought.evaluation_previous_goal,
                "memory": thought.memory,
                "next_goal": thought.next_goal,
            }
            f.write(json.dumps(log_entry) + "\n")

    my_step_hook.last_logged_index += len(new_thoughts)


async def main():
    agent = Agent(
        llm=teacher_llm,
        task=(
            "Create an account on outlook mail abcbddasdas at some domain. "
            "If you're stuck or need inspiration, stop running."
        ),
        browser_session=browser_session,
        save_conversation_path="conversation.jsonl",
        use_vision=False,
    )

    result = await agent.run(on_step_end=my_step_hook)
    print("Final Result from Teacher Agent:\n", result)


if __name__ == "__main__":
    asyncio.run(main())
