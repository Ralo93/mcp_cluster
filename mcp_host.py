from smolagents.mcp_client import MCPClient
from smolagents import CodeAgent, InferenceClientModel
import gradio as gr

try:
    # Connect to local MCP server
    mcp_client = MCPClient({"url": "http://127.0.0.1:7860/"})
    
    # Load the tools registered by the MCP server
    tools = mcp_client.get_tools()
    
    # Load an inference model and create the agent with the tools
    model = InferenceClientModel()
    agent = CodeAgent(tools=tools, model=model)

    # Define the interaction
    def chat_fn(message, history):
        return str(agent.run(message))

    # Set up Gradio ChatInterface
    demo = gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        examples=["Query speeches about climate change", "Find speeches by Merkel about energy"],
        title="MCP Agent",
        description="Talk to an agent that uses local tools from your Gradio MCP server.",
    )

    # Launch the host interface
    demo.launch()
finally:
    mcp_client.close()
