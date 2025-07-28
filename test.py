import json
import asyncio
from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm import ChatGoogle

def load_teacher_thoughts(path="thoughts_log.jsonl"):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def format_teacher_context(teacher_thoughts, count=3):
    selected = teacher_thoughts[:count]
    blocks = []
    for i, t in enumerate(selected):
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
    if not hasattr(student_step_hook, "teacher_thoughts"):
        student_step_hook.teacher_thoughts = load_teacher_thoughts()

    teacher_thoughts = student_step_hook.teacher_thoughts
    student_thoughts = agent.state.history.model_thoughts()
    current_step = len(student_thoughts) - 1

    if current_step < len(teacher_thoughts):
        t = teacher_thoughts[current_step]
        print(f"[STUDENT] Learning from TEACHER Step {current_step}")
        print("👓 Teacher Thinking:", t.get("thinking"))
        print("🧠 Student should reflect similarly...")

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

async def main_student():
    teacher_thoughts = load_teacher_thoughts()

    student_llm = ChatGoogle(model='gemini-2.0-flash')

    browser_profile_student = BrowserProfile(
        headless=True,
        viewport={"width": 1280, "height": 1100},
        user_data_dir="/home/drunkencloud/work/maheswari/testing-browser-use/student-dir",
    )

    browser_session_student = BrowserSession(
        browser_profile=browser_profile_student,
    )

    system_prompt = format_teacher_context(teacher_thoughts, count=3)

    student_agent = Agent(
        llm=student_llm,
        task="Get me the first line from the wikipedia article on flowers (https://en.wikipedia.org/wiki/Flower). If you are stuck on a captcha, stop running.",
        system_prompt=system_prompt,
        save_conversation_path='student_conversation.jsonl',
        use_vision=False,
        browser_session=browser_session_student,
    )

    result = await student_agent.run(
        on_step_end=student_step_hook
    )

    print("🎓 Student Agent Result:", result)

if __name__ == "__main__":
    asyncio.run(main_student())
