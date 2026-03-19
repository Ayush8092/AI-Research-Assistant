"""
Knowledge Graph Builder

Fixes:
  - Added CT, MRI, X-ray, mammography as dataset nodes
  - Better paper labels (AuthorYear format)
  - Cross-paper edges for shared methods/datasets
  - Cleaner visualization with legend
"""

import json
import logging
import re
from pathlib import Path
from agents.state import ResearchState
import networkx as nx

logger = logging.getLogger(__name__)

METHOD_KEYWORDS = [
    "transformer", "bert", "gpt", "llm", "lstm", "cnn",
    "convolutional neural network", "resnet", "vgg", "densenet",
    "u-net", "vit", "vision transformer", "random forest", "svm",
    "xgboost", "attention", "federated learning", "diffusion model",
    "gan", "vae", "reinforcement learning", "fine-tuning",
    "transfer learning", "deep learning", "neural network",
    "contrastive learning", "knowledge distillation"
]

DATASET_KEYWORDS = [
    # Medical imaging modalities — important fix
    "ct scan", "mri", "x-ray", "mammography", "ultrasound",
    "pet scan", "histology", "dermoscopy", "fundus imaging",
    # Named medical datasets
    "mimic", "chexpert", "imagenet", "physionet", "ukbiobank",
    "nih chest", "luna16", "lidc", "brats", "isic", "kits",
    "covid-19", "montgomery", "ddsm", "vindr", "medmnist", "tcia",
    # NLP/ML datasets
    "squad", "glue", "mmlu", "coco", "vqa"
]

TASK_KEYWORDS = [
    "cancer detection", "tumor detection", "lesion detection",
    "image segmentation", "image classification", "image registration",
    "diagnosis", "prognosis", "risk prediction",
    "object detection", "semantic segmentation",
    "clinical decision support", "drug discovery",
    "question answering", "text generation",
    "named entity recognition", "sentiment analysis"
]


def knowledge_graph_agent(state: ResearchState) -> ResearchState:
    """LangGraph node: KnowledgeGraphAgent."""
    ranked_papers = state.get("ranked_papers", [])

    if not ranked_papers:
        return {**state,
                "knowledge_graph_entities": [],
                "knowledge_graph_edges":    []}

    logger.info(f"[KGAgent] Building graph from {len(ranked_papers)} papers...")

    entities           = []
    edges              = []
    G                  = nx.DiGraph()
    method_users:  dict[str, list[str]] = {}
    dataset_users: dict[str, list[str]] = {}

    for paper in ranked_papers[:15]:
        pid      = paper["paper_id"]
        title    = paper.get("title","")
        authors  = paper.get("authors",[])
        year     = paper.get("published","")[:4]
        abstract = paper.get("abstract","")
        insights = paper.get("insights",{})

        full_text = (
            title + " " + abstract + " " +
            insights.get("methodology","") + " " +
            insights.get("datasets","")
        ).lower()

        short_label = _make_label(authors, year, title)
        G.add_node(pid, type="paper", label=short_label,
                   full_title=title[:80], year=year)
        entities.append({"id": pid, "type": "paper",
                         "name": short_label, "full_title": title[:80]})

        # Methods
        for method in _extract_entities(full_text, METHOD_KEYWORDS):
            mid = f"method:{method}"
            if not G.has_node(mid):
                G.add_node(mid, type="method",
                           label=method.title()[:25])
                entities.append({"id": mid, "type": "method",
                                  "name": method.title()})
            G.add_edge(pid, mid, relation="uses")
            edges.append({"source": pid, "relation": "uses", "target": mid})
            method_users.setdefault(mid, []).append(pid)

        # Datasets — now includes CT/MRI/etc
        for ds in _extract_entities(full_text, DATASET_KEYWORDS):
            did = f"dataset:{ds}"
            if not G.has_node(did):
                display = ds.upper() if len(ds) <= 6 else ds.title()
                G.add_node(did, type="dataset", label=display[:25])
                entities.append({"id": did, "type": "dataset",
                                  "name": display})
            G.add_edge(pid, did, relation="trained_on")
            edges.append({"source": pid, "relation": "trained_on",
                          "target": did})
            dataset_users.setdefault(did, []).append(pid)

        # Tasks
        for task in _extract_entities(full_text, TASK_KEYWORDS):
            tid = f"task:{task}"
            if not G.has_node(tid):
                G.add_node(tid, type="task", label=task.title()[:25])
                entities.append({"id": tid, "type": "task",
                                  "name": task.title()})
            G.add_edge(pid, tid, relation="solves")
            edges.append({"source": pid, "relation": "solves", "target": tid})

    # Cross-paper edges: shared methods
    cross = 0
    for mid, users in method_users.items():
        if len(users) >= 2:
            method_name = mid.replace("method:", "")
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    p1, p2 = users[i], users[j]
                    if not G.has_edge(p1, p2):
                        G.add_edge(p1, p2, relation=f"shares {method_name}")
                        edges.append({"source": p1,
                                      "relation": f"shares {method_name}",
                                      "target": p2})
                        cross += 1

    # Cross-paper edges: shared datasets
    for did, users in dataset_users.items():
        if len(users) >= 2:
            ds_name = did.replace("dataset:", "")
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    p1, p2 = users[i], users[j]
                    if not G.has_edge(p1, p2):
                        G.add_edge(p1, p2, relation=f"shares {ds_name}")
                        edges.append({"source": p1,
                                      "relation": f"shares {ds_name}",
                                      "target": p2})
                        cross += 1

    # Deduplicate
    seen, unique_edges = set(), []
    for e in edges:
        key = (e["source"], e["relation"], e["target"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    logger.info(f"[KGAgent] {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges "
                f"({cross} cross-paper)")

    html_path = _export_viz(G)

    return {
        **state,
        "knowledge_graph_entities": entities,
        "knowledge_graph_edges":    unique_edges,
        "artifacts": {
            **state.get("artifacts",{}),
            "knowledge_graph_html": html_path
        }
    }


def _make_label(authors: list, year: str, title: str) -> str:
    """Short readable label: LastName YEAR."""
    if authors:
        last = authors[0].split()[-1] if authors[0].split() else "Unknown"
        last = re.sub(r"[^a-zA-Z]", "", last)[:12]
        return f"{last} {year}"
    words = title.split()[:2]
    return " ".join(words)[:15] + ".."


def _extract_entities(text: str, keywords: list[str]) -> list[str]:
    return [kw for kw in keywords if kw in text]


def _export_viz(G: nx.DiGraph) -> str:
    output_dir = Path("data/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path  = str(output_dir / "knowledge_graph.html")

    try:
        from pyvis.network import Network

        net = Network(height="600px", width="100%",
                      directed=True, notebook=False)
        net.force_atlas_2based(gravity=-50, spring_length=120)

        colors = {"paper":"#4A90D9","method":"#E67E22",
                  "dataset":"#27AE60","task":"#9B59B6"}
        sizes  = {"paper":25,"method":15,"dataset":15,"task":12}

        for node_id, attrs in G.nodes(data=True):
            ntype      = attrs.get("type","paper")
            label      = attrs.get("label", str(node_id)[:20])
            full_title = attrs.get("full_title", label)
            net.add_node(
                str(node_id),
                label=label,
                color=colors.get(ntype,"#888"),
                title=f"<b>{ntype.upper()}</b><br>{full_title}",
                size=sizes.get(ntype,12),
                font={"size":11,"color":"#ffffff"}
            )

        for src, dst, attrs in G.edges(data=True):
            rel      = attrs.get("relation","")
            is_cross = "shares" in rel
            net.add_edge(
                str(src), str(dst),
                label=rel[:18],
                color="#ff6b6b" if is_cross else "#888888",
                dashes=is_cross,
                width=2 if is_cross else 1,
                arrows="to"
            )

        legend = """
        <div style="position:absolute;top:10px;right:10px;
                    background:rgba(0,0,0,0.85);padding:12px;
                    border-radius:8px;font-size:12px;color:white;
                    font-family:Arial;">
            <b>Legend</b><br>
            <span style="color:#4A90D9">●</span> Paper<br>
            <span style="color:#E67E22">●</span> Method<br>
            <span style="color:#27AE60">●</span> Dataset<br>
            <span style="color:#9B59B6">●</span> Task<br>
            <span style="color:#ff6b6b">---</span> Shared
        </div>"""
        net.html = net.html.replace("</body>", f"{legend}</body>")
        net.save_graph(html_path)
        logger.info(f"[KGAgent] Saved: {html_path}")

    except ImportError:
        json_path = str(Path("data/artifacts") / "knowledge_graph.json")
        with open(json_path,"w") as f:
            json.dump({
                "nodes": [{"id":n,**G.nodes[n]} for n in G.nodes()],
                "edges": [{"source":u,"target":v,**d}
                          for u,v,d in G.edges(data=True)]
            }, f, indent=2)
        return json_path
    except Exception as e:
        logger.error(f"[KGAgent] Viz failed: {e}")
        return ""

    return html_path


def get_graph_stats(entities, edges) -> dict:
    tc, rc = {}, {}
    for e in entities:
        t = e.get("type","unknown")
        tc[t] = tc.get(t,0) + 1
    for edge in edges:
        r    = edge.get("relation","unknown")
        rkey = "cross_paper" if "shares" in r else r
        rc[rkey] = rc.get(rkey,0) + 1
    return {
        "total_entities": len(entities),
        "total_edges":    len(edges),
        "entity_types":   tc,
        "relation_types": rc
    }