import json
import asyncio
from browser_use import Agent, BrowserProfile, BrowserSession, Controller, ActionResult
from browser_use.llm import ChatGoogle


controller = Controller()

@controller.action("Ask the teacher agent how they approached this task when stuck")
def ask_teacher_context() -> ActionResult:
    try:
        with open("thoughts_log.jsonl", "r") as f:
            teacher_thoughts = [json.loads(line) for line in f]
    except FileNotFoundError:
        return ActionResult(extracted_content="The teacher's guidance is not available.", include_in_memory=True)

    blocks = []
    for i, t in enumerate(teacher_thoughts):
        blocks.append(
            f"""Step {i+1}:
Thinking: {t.get("thinking")}
Evaluation: {t.get("evaluation_previous_goal")}
Memory: {t.get("memory")}
Next Goal: {t.get("next_goal")}\n"""
        )

    content = "Here is what the teacher did:\n\n" + "\n".join(blocks)
    print(blocks)
    return ActionResult(extracted_content=content, include_in_memory=True)


async def run_student_agent_with_controller(llm, session):
    agent = Agent(
        llm=llm,
        task=(
            "Create an account on outlook mail abcbddasdas at some domain. If you're stuck or need inspiration, you may call ask_teacher_context() to see how a smarter agent approached similar problems. If it doesnt solve your issue, stop running"
        ),
        controller=controller,
        browser_session=session,
        save_conversation_path="student_conversation.jsonl",
        use_vision=False,
    )
    result = await agent.run()
    print("Final result from the student agent:\n", result)


async def main():
    profile = BrowserProfile(headless=False)
    session = BrowserSession(browser_profile=profile)
    llm = ChatGoogle(model="gemini-2.0-flash")

    await run_student_agent_with_controller(llm, session)

if __name__ == "__main__":
    asyncio.run(main())
