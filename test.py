import json
import asyncio
from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm import ChatGoogle

def load_teacher_thoughts(path="thoughts_log.jsonl"):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def format_teacher_context(teacher_thoughts):
    blocks = []
    for i, t in enumerate(teacher_thoughts):
        blocks.append(
            f"""Step {i}:
Thinking: {t.get("thinking")}
Evaluation: {t.get("evaluation_previous_goal")}
Memory: {t.get("memory")}
Next Goal: {t.get("next_goal")}\n"""
        )
    return (
        "You are a student agent learning from a smarter teacher agent. "
        "Here is how the teacher approached a similar task in their first few steps:\n\n"
        + "\n".join(blocks)
        + "\nUse these examples to guide your own reasoning."
    )

async def student_step_hook(agent):
    student_thoughts = agent.state.history.model_thoughts()
    latest_thought = student_thoughts[-1] if student_thoughts else None
    if latest_thought:
        log_entry = {
            "thinking": latest_thought.thinking,
            "evaluation_previous_goal": latest_thought.evaluation_previous_goal,
            "memory": latest_thought.memory,
            "next_goal": latest_thought.next_goal,
        }
        with open("student_thoughts.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

def get_browser_session():
    profile = BrowserProfile(
        headless=True,
        viewport={"width": 1280, "height": 1100},
        user_data_dir=None
    )
    return BrowserSession(browser_profile=profile)

async def run_student_with_extend_system_message(llm, session, instructions):
    agent = Agent(
        llm=llm,
        task="Get me the first line from the wikipedia article on flowers (https://en.wikipedia.org/wiki/Flower). If you are stuck on a captcha, stop running.",
        extend_system_message=instructions,
        save_conversation_path='student_conversation_extend.jsonl',
        use_vision=False,
        browser_session=session,
    )
    result = await agent.run(on_step_end=student_step_hook)
    print("Result with `extend_system_message`:\n", result)

async def run_student_with_message_context(llm, session, instructions):
    agent = Agent(
        llm=llm,
        task="Get me the first line from the wikipedia article on flowers (https://en.wikipedia.org/wiki/Flower). If you are stuck on a captcha, stop running.",
        message_context=instructions,
        save_conversation_path='student_conversation_message.jsonl',
        use_vision=False,
        browser_session=session,
    )
    result = await agent.run(on_step_end=student_step_hook)
    print("Result with `message_context`:\n", result)

async def run_student_with_override_system_message(llm, session, instructions):
    agent = Agent(
        llm=llm,
        task="Get me the first line from the wikipedia article on flowers (https://en.wikipedia.org/wiki/Flower). If you are stuck on a captcha, stop running.",
        override_system_message=instructions,
        save_conversation_path='student_conversation_override.jsonl',
        use_vision=False,
        browser_session=session,
    )
    result = await agent.run(on_step_end=student_step_hook)
    print("Result with `override_system_message`:\n", result)

async def main_student():
    teacher_thoughts = load_teacher_thoughts()
    instructions = format_teacher_context(teacher_thoughts)
    llm = ChatGoogle(model='gemini-2.0-flash')
    session = get_browser_session()

    await run_student_with_extend_system_message(llm, session, instructions)
    await run_student_with_message_context(llm, session, instructions)
    await run_student_with_override_system_message(llm, session, instructions)

if __name__ == "__main__":
    asyncio.run(main_student())