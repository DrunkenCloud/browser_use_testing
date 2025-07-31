import json
from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm import ChatGoogle
from dotenv import load_dotenv
import asyncio

load_dotenv()

teacher_llm = ChatGoogle(model='gemini-2.0-flash-exp')

browser_profile = BrowserProfile(
    headless=True,
    viewport={"width": 1280, "height": 1100},
    user_data_dir=None,
)

browser_session = BrowserSession(browser_profile=browser_profile)


async def my_step_hook(agent):
    output_path = "thoughts_log.jsonl"

    if not hasattr(my_step_hook, "last_saved_index"):
        my_step_hook.last_saved_index = 0

    model_thoughts = agent.state.history.model_thoughts()
    new_thoughts = model_thoughts[my_step_hook.last_saved_index:]

    with open(output_path, "a") as f:
        for thought in new_thoughts:
            log_entry = {
                "thinking": thought.thinking,
                "evaluation_previous_goal": thought.evaluation_previous_goal,
                "memory": thought.memory,
                "next_goal": thought.next_goal,
            }
            f.write(json.dumps(log_entry) + "\n")


    my_step_hook.last_saved_index += len(new_thoughts)


async def main():
    agent = Agent(
        llm=teacher_llm,
        task="Get me the first line from the wikipedia article on flowers (https://en.wikipedia.org/wiki/Flower). If you are stuck on a captcha, stop running.",
        save_conversation_path='conversation.jsonl',
        use_vision=False,
        browser_session=browser_session,
    )

    result = await agent.run(on_step_end=my_step_hook)

    print("Agent result:", result)


if __name__ == "__main__":
    asyncio.run(main())
