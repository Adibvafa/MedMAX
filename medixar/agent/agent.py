from typing import List, Dict, Any, TypedDict, Annotated
import operator
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

_ = load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that can perform actions based on language model responses.

    Attributes:
        system (str): The system message for the agent.
        graph (StateGraph): The compiled state graph for the agent's workflow.
        tools (Dict[str, BaseTool]): A dictionary of available tools for the agent.
        model (BaseLanguageModel): The language model used by the agent.
    """

    def __init__(
        self, model: BaseLanguageModel, tools: List[BaseTool], system: str = ""
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of tools available to the agent.
            system (str, optional): The system message. Defaults to "".
        """
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("assisstant", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "assisstant", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "assisstant")
        graph.set_entry_point("assisstant")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState) -> bool:
        """
        Check if there are any tool calls in the last message.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if there are tool calls, False otherwise.
        """
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Invoke the language model with the current messages.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls based on the language model's response.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing the results of tool executions.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t["name"] not in self.tools:  # check for bad tool name from assisstant
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct assisstant to retry if bad
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}
