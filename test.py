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
    return ActionResult(extracted_content=content, include_in_memory=True)


@controller.action("Launch a better agent to try the task when teacher guidance is insufficient")
def ask_better_agent_for_help() -> ActionResult:
    async def run_better_agent():
        better_llm = ChatGoogle(model="gemini-2.0-flash")
        incognito_profile = BrowserProfile(
            headless=False,
            user_data_dir=None,
            viewport={"width": 1280, "height": 1100},
        )
        incognito_session = BrowserSession(browser_profile=incognito_profile)

        better_agent = Agent(
            llm=better_llm,
            task="Try to create an Outlook account with the ID 'abcbddasdas'. Try harder and smarter.",
            browser_session=incognito_session,
            save_conversation_path="better_agent_conversation.jsonl",
            use_vision=False,
        )

        result = await better_agent.run()
        return result

    try:
        result = asyncio.run(run_better_agent())
        return ActionResult(extracted_content=f"The better agent tried and got:\n\n{result}", include_in_memory=True)
    except Exception as e:
        return ActionResult(extracted_content=f"Failed to run better agent: {e}", include_in_memory=True)


async def run_student_agent_with_controller(llm, session):
    agent = Agent(
        llm=llm,
        task=(
            "Create an account on Outlook mail with the ID 'abcbddasdas'. "
            "If you're stuck or need inspiration, you may call ask_teacher_context() to see how a smarter agent approached similar problems. "
            "If that doesn't help, call ask_better_agent_for_help() to try using a more powerful agent."
        ),
        controller=controller,
        browser_session=session,
        save_conversation_path="student_conversation.jsonl",
        use_vision=False,
    )
    result = await agent.run()
    print("Final result from the student agent:\n", result)


async def main():
    incognito_profile = BrowserProfile(
        headless=False,
        user_data_dir=None,
        viewport={"width": 1280, "height": 1100},
    )
    session = BrowserSession(browser_profile=incognito_profile)
    llm = ChatGoogle(model="gemini-2.0-flash")

    await run_student_agent_with_controller(llm, session)

if __name__ == "__main__":
    asyncio.run(main())
