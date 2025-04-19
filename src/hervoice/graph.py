from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledGraph
from langchain_core.runnables.graph import MermaidDrawMethod

from .agent.state import ChatState
from .agent.web_searcher import search_web
from .agent.document_retriever import retrieve_documents
from .agent.chitter_chatter import chitter_chatter_agent
from .agent.query_rewriter import rewrite_query
from .agent.evaluator import evaluate_answer_by_design
from .agent.route_question import query_router_agent
from .agent.answer_generator import answer_generator
from .agent.reference_table import create_reference_table
from .agent.answer_verifier import check_relevance

def save_graph_as_png(app: CompiledGraph, output_file_path) -> None:
    png_image = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    file_path = output_file_path if len(output_file_path) > 0 else "graph.png"
    with open(file_path, "wb") as f:
        f.write(png_image)
    return file_path


def build_rag_graph(selected_analysts):
    # === Initialize Graph ===
    builder = StateGraph(ChatState)

    # === Add Agent Nodes ===
    builder.add_node("web_searcher", search_web)
    builder.add_node("document_retriever", retrieve_documents)
    builder.add_node("chitter_chatter", chitter_chatter_agent)
    builder.add_node("query_rewriter", rewrite_query)
    builder.add_node("evaluate_answer", evaluate_answer_by_design)
    builder.add_node("answer_generator", answer_generator)
    builder.add_node("relevance_grader", check_relevance)

    # === Entry point: Route query to appropriate agent ===
    builder.set_conditional_entry_point(
        query_router_agent,
        {
            "Websearch": "web_searcher",
            "Vectorstore": "document_retriever",
            "Chitter-Chatter": "chitter_chatter",
        },
    )
    builder.add_edge("document_retriever", "relevance_grader")
    builder.add_edge("web_searcher", "answer_generator")
    builder.add_edge("web_searcher", "answer_generator")

    builder.add_conditional_edges(
        "relevance_grader",
        check_relevance,
        path_map={
            "Websearch": "web_searcher",
            "generate": "answer_generator",
        },
    )
    builder.add_edge("query_rewriter", "document_retriever")

    builder.add_conditional_edges(
        "evaluate_answer",
        evaluate_answer_by_design,
        path_map={
            "useful": END,
            "design-P1-failed": "query_rewriter",
            "design-P2-failed": "query_rewriter",
            "design-P3-failed": "query_rewriter",
            "hallucination-checker-failed": "query_rewriter",
            "max-retries": "chitter_chatter"
        }

    )

    builder.add_edge("chitter_chatter", END)
    return builder

    # builder.add_node("generate_answer", generate_answer)

    # builder.add_node("check_hallucination", check_hallucination)

    # builder.add_edge("generate_answer", "check_hallucination")
    # builder.add_edge("retry_generation", "check_hallucination")
    # builder.add_conditional_edges(
    #     "check_hallucination",
    #     path_map={
    #         "check_relevance": check_relevance,
    #         "retry_generation": generate_answer,
    #         "hallucinated": chitter_chatter_agent,
    #     }
    # )

    # builder.add_conditional_edges(
    #     "check_relevance",
    #     path_map={
    #         "useful": END,
    #         "not_useful": chitter_chatter_agent,
    #     }
    # )
    # builder.add_edge("chitter_chatter", END)

    # return builder.compile()


# rag_graph = build_rag_graph()
