import json
import asyncio
from typing import Dict, Any
from browser_use import Agent, BrowserProfile, BrowserSession, Controller, ActionResult
from browser_use.llm import ChatGoogle

controller = Controller()

task_contexts = {}

class TaskContext:
    def __init__(self, task: str, session_id: str = None):
        self.task = task
        self.original_task = task
        self.session_id = session_id or str(id(self))


@controller.action("Ask the teacher agent how they approached this task when stuck")
def ask_teacher_context(context: Any = None) -> ActionResult:
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
def ask_better_agent_for_help(context: Any = None) -> ActionResult:
    async def run_better_agent(original_task: str, session_id: str):
        better_llm = ChatGoogle(model="gemini-2.0-flash")
        incognito_profile = BrowserProfile(
            headless=False,
            user_data_dir=None,
            viewport={"width": 1280, "height": 1100},
        )
        incognito_session = BrowserSession(browser_profile=incognito_profile)

        enhanced_task = f"{original_task} Try harder and smarter with different approaches if the standard method fails."

        better_agent = Agent(
            llm=better_llm,
            task=enhanced_task,
            browser_session=incognito_session,
            save_conversation_path=f"better_agent_conversation_{session_id}.jsonl",
            use_vision=False,
        )

        result = await better_agent.run()
        return result

    try:
        original_task = None
        session_id = "unknown"
        
        if context:
            if hasattr(context, 'task'):
                original_task = context.task
                session_id = getattr(context, 'session_id', session_id)
            elif hasattr(context, 'original_task'):
                original_task = context.original_task
                session_id = getattr(context, 'session_id', session_id)
            elif isinstance(context, dict):
                original_task = context.get('task') or context.get('original_task')
                session_id = context.get('session_id', session_id)
        
        if not original_task and context and hasattr(context, 'session_id'):
            stored_context = task_contexts.get(context.session_id)
            if stored_context:
                original_task = stored_context.original_task
                session_id = stored_context.session_id
        
        if not original_task:
            return ActionResult(
                extracted_content="No task available for the better agent to work on.", 
                include_in_memory=True
            )
        
        result = asyncio.run(run_better_agent(original_task, session_id))
        return ActionResult(
            extracted_content=f"The better agent tried and got:\n\n{result}", 
            include_in_memory=True
        )
    except Exception as e:
        return ActionResult(
            extracted_content=f"Failed to run better agent: {e}", 
            include_in_memory=True
        )


async def run_student_agent_with_controller(llm, session, task: str, session_id: str = None):
    task_context = TaskContext(task, session_id)
    
    task_contexts[task_context.session_id] = task_context
    
    enhanced_task = (
        f"{task} "
        "If you're stuck or need inspiration, you may call ask_teacher_context() to see how a smarter agent approached similar problems. "
        "If that doesn't help, call ask_better_agent_for_help() to try using a more powerful agent."
    )
    
    try:
        agent = Agent(
            llm=llm,
            task=enhanced_task,
            controller=controller,
            browser_session=session,
            context=task_context,
            save_conversation_path=f"student_conversation_{task_context.session_id}.jsonl",
            use_vision=False,
        )
        result = await agent.run()
        print(f"Final result from the student agent (session {task_context.session_id}):\n", result)
        return result
    finally:
        task_contexts.pop(task_context.session_id, None)


async def main():
    incognito_profile = BrowserProfile(
        headless=False,
        user_data_dir=None,
        viewport={"width": 1280, "height": 1100},
    )
    session = BrowserSession(browser_profile=incognito_profile)
    llm = ChatGoogle(model="gemini-2.0-flash")

    main_task = "Create an account on Outlook mail with the ID 'abcbddasdas'."
    
    await run_student_agent_with_controller(llm, session, main_task)


if __name__ == "__main__":
    asyncio.run(main())