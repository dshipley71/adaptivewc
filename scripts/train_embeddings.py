#!/usr/bin/env python3
"""
Train and use ML embeddings with crawled page structures.

This script demonstrates how to:
1. Export structure data for ML training
2. Create embeddings using sentence-transformers
3. Find similar page structures
4. Train a page type classifier (LogisticRegression, XGBoost, or LightGBM)
5. Detect changes using ML-based change detection
6. Generate descriptions using rules or LLM (OpenAI, Anthropic, Ollama)

Usage:
    # Export training data from Redis
    python scripts/train_embeddings.py export -o training_data.jsonl

    # Create embeddings for all structures
    python scripts/train_embeddings.py embed

    # Find similar structures to a domain
    python scripts/train_embeddings.py similar example.com

    # Train a page type classifier (supports --classifier-type)
    python scripts/train_embeddings.py train --classifier-type xgboost

    # Predict page type for a new structure
    python scripts/train_embeddings.py predict example.com --classifier classifier.pkl

    # Set baseline and detect changes
    python scripts/train_embeddings.py set-baseline example.com
    python scripts/train_embeddings.py detect-drift example.com

    # Generate descriptions with rules or LLM
    python scripts/train_embeddings.py describe example.com --mode rules
    python scripts/train_embeddings.py describe example.com --mode llm --provider openai
    python scripts/train_embeddings.py describe example.com --mode llm --provider anthropic
    python scripts/train_embeddings.py describe example.com --mode llm --provider ollama
    python scripts/train_embeddings.py describe example.com --mode llm --provider ollama --llm-model llama3.2
    python scripts/train_embeddings.py describe example.com --mode llm --provider ollama --ollama-url http://192.168.1.100:11434
    python scripts/train_embeddings.py describe example.com --mode llm --provider ollama-cloud

Requirements:
    pip install sentence-transformers scikit-learn

    For XGBoost:      pip install xgboost
    For LightGBM:     pip install lightgbm  (already in main deps)
    For LLM mode:     pip install openai anthropic  (for OpenAI/Anthropic)
    For Ollama:       No extra deps (uses httpx, already in main deps)

Environment Variables:
    OPENAI_API_KEY     - Required for --provider openai
    ANTHROPIC_API_KEY  - Required for --provider anthropic
    OLLAMA_BASE_URL    - Custom Ollama URL (default: http://localhost:11434)
    OLLAMA_API_KEY     - Required for --provider ollama-cloud
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import redis.asyncio as redis

# Add crawler to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler.config import load_config
from crawler.storage.structure_store import StructureStore
from crawler.ml.embeddings import (
    ClassifierType,
    DescriptionMode,
    StructureEmbeddingModel,
    StructureClassifier,
    StructureDescriptionGenerator,
    MLChangeDetector,
    get_description_generator,
    export_training_data,
    create_similarity_pairs,
)


async def get_all_structures(structure_store: StructureStore):
    """Get all structures and strategies from Redis."""
    domains = await structure_store.list_domains()
    structures = []
    strategies = []

    for domain, page_type in domains:
        structure = await structure_store.get_structure(domain, page_type)
        strategy = await structure_store.get_strategy(domain, page_type)

        if structure:
            structures.append(structure)
            strategies.append(strategy)

    return structures, strategies


async def cmd_export(args, structure_store: StructureStore):
    """Export training data to JSONL."""
    print(f"Exporting training data to {args.output}...")

    structures, strategies = await get_all_structures(structure_store)

    if not structures:
        print("No structures found in Redis.")
        return

    # Filter out structures without strategies
    pairs = [(s, st) for s, st in zip(structures, strategies) if st is not None]
    if pairs:
        structures, strategies = zip(*pairs)
        export_training_data(list(structures), list(strategies), args.output)
        print(f"Exported {len(structures)} structure-strategy pairs.")
    else:
        # Export structures only
        desc_gen = StructureDescriptionGenerator()
        with open(args.output, "w") as f:
            for structure in structures:
                record = {
                    "text": desc_gen.generate(structure),
                    "label": structure.page_type,
                    "domain": structure.domain,
                }
                f.write(json.dumps(record) + "\n")
        print(f"Exported {len(structures)} structures (no strategies).")


async def cmd_embed(args, structure_store: StructureStore):
    """Create embeddings for all structures."""
    print("Creating embeddings...")

    structures, _ = await get_all_structures(structure_store)

    if not structures:
        print("No structures found in Redis.")
        return

    model = StructureEmbeddingModel(model_name=args.model)
    print(f"Using model: {args.model}")

    embeddings = model.embed_structures_batch(structures)

    # Save embeddings
    output = args.output or "embeddings.json"
    with open(output, "w") as f:
        json.dump([e.to_dict() for e in embeddings], f, indent=2)

    print(f"Created {len(embeddings)} embeddings.")
    print(f"Saved to: {output}")

    # Show sample
    print("\nSample embedding:")
    print(f"  Domain: {embeddings[0].domain}")
    print(f"  Description: {embeddings[0].text_description[:100]}...")
    print(f"  Embedding dims: {len(embeddings[0].embedding)}")


async def cmd_similar(args, structure_store: StructureStore):
    """Find structures similar to a domain."""
    print(f"Finding structures similar to {args.domain}...")

    structures, _ = await get_all_structures(structure_store)

    if not structures:
        print("No structures found in Redis.")
        return

    # Find the query structure
    query_structure = None
    for s in structures:
        if s.domain == args.domain:
            query_structure = s
            break

    if query_structure is None:
        print(f"Structure for {args.domain} not found.")
        return

    model = StructureEmbeddingModel(model_name=args.model)

    # Create all embeddings
    print("Creating embeddings...")
    embeddings = model.embed_structures_batch(structures)

    # Find query embedding
    query_embedding = None
    for e in embeddings:
        if e.domain == args.domain:
            query_embedding = e
            break

    # Find similar
    results = model.find_similar(
        query_embedding.embedding,
        [e for e in embeddings if e.domain != args.domain],
        top_k=args.top_k,
    )

    print(f"\nTop {args.top_k} similar structures to {args.domain}:")
    print("=" * 60)
    for emb, score in results:
        print(f"\n  {emb.domain} [{emb.page_type}]")
        print(f"  Similarity: {score:.2%}")
        print(f"  Description: {emb.text_description[:80]}...")


async def cmd_train_classifier(args, structure_store: StructureStore):
    """Train a page type classifier."""
    classifier_type = ClassifierType(args.classifier_type)
    print(f"Training page type classifier using {classifier_type.value}...")

    structures, _ = await get_all_structures(structure_store)

    if len(structures) < 5:
        print("Need at least 5 structures to train a classifier.")
        return

    # Use page_type as labels
    labels = [s.page_type for s in structures]

    # Check we have multiple classes
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        print(f"Need at least 2 different page types. Found: {unique_labels}")
        return

    model = StructureEmbeddingModel(model_name=args.model)
    classifier = StructureClassifier(model, classifier_type=classifier_type)

    print(f"Training on {len(structures)} structures...")
    print(f"Page types: {unique_labels}")

    metrics = classifier.train(structures, labels)

    print("\nTraining Results:")
    print(f"  Classifier: {metrics.get('classifier_type', 'logistic_regression')}")
    print(f"  Accuracy: {metrics['accuracy']:.2%} (+/- {metrics['std']:.2%})")
    print(f"  Samples: {metrics['num_samples']}")
    print(f"  Classes: {metrics['num_classes']}")

    # Feature importance for tree-based classifiers
    importance = classifier.get_feature_importance()
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 Feature Importance:")
        for feat, imp in sorted_imp:
            print(f"  {feat}: {imp:.4f}")

    # Save classifier
    output = args.output or "classifier.pkl"
    classifier.save(output)
    print(f"\nClassifier saved to: {output}")


async def cmd_predict(args, structure_store: StructureStore):
    """Predict page type for a domain."""
    if not args.classifier:
        print("Specify --classifier path to load trained model.")
        return

    structure = await structure_store.get_structure(args.domain, args.page_type or "homepage")
    if structure is None:
        # Try to find any page type
        domains = await structure_store.list_domains()
        for d, pt in domains:
            if d == args.domain:
                structure = await structure_store.get_structure(d, pt)
                break

    if structure is None:
        print(f"No structure found for {args.domain}")
        return

    model = StructureEmbeddingModel(model_name=args.model)
    classifier = StructureClassifier(model)
    classifier.load(args.classifier)

    label, confidence = classifier.predict(structure)

    print(f"\nPrediction for {structure.domain}:")
    print(f"  Actual page type: {structure.page_type}")
    print(f"  Predicted: {label}")
    print(f"  Confidence: {confidence:.2%}")


async def cmd_similarity_pairs(args, structure_store: StructureStore):
    """Create similarity pairs for contrastive learning."""
    print("Creating similarity pairs for fine-tuning...")

    structures, _ = await get_all_structures(structure_store)

    if len(structures) < 2:
        print("Need at least 2 structures.")
        return

    pairs = create_similarity_pairs(structures)

    output = args.output or "similarity_pairs.jsonl"
    with open(output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Created {len(pairs)} similarity pairs.")
    print(f"Saved to: {output}")

    # Show sample
    if pairs:
        print("\nSample pair:")
        print(f"  Text 1: {pairs[0]['sentence1'][:60]}...")
        print(f"  Text 2: {pairs[0]['sentence2'][:60]}...")
        print(f"  Score: {pairs[0]['score']}")


async def cmd_describe(args, structure_store: StructureStore):
    """Generate description of a structure."""
    structure = await structure_store.get_structure(args.domain, args.page_type or "homepage")
    if structure is None:
        # Try to find any page type
        domains = await structure_store.list_domains()
        for d, pt in domains:
            if d == args.domain:
                structure = await structure_store.get_structure(d, pt)
                break

    if structure is None:
        print(f"No structure found for {args.domain}")
        return

    mode = DescriptionMode(args.mode)
    print(f"Generating description using {mode.value} mode...")

    if mode == DescriptionMode.LLM:
        kwargs = {
            "provider": args.provider,
            "model": args.llm_model,
        }
        # Add ollama_base_url if specified
        if hasattr(args, "ollama_url") and args.ollama_url:
            kwargs["ollama_base_url"] = args.ollama_url
        generator = get_description_generator(mode, **kwargs)
        print(f"Using provider: {args.provider}" + (f" (model: {args.llm_model})" if args.llm_model else ""))
    else:
        generator = get_description_generator(mode)

    description = generator.generate(structure)

    print(f"\nStructure for {structure.domain} [{structure.page_type}]:")
    print("=" * 60)
    print(description)


async def cmd_set_baseline(args, structure_store: StructureStore, detector: MLChangeDetector):
    """Set baseline for a domain."""
    structure = await structure_store.get_structure(args.domain, args.page_type or "homepage")
    if structure is None:
        # Try to find any page type
        domains = await structure_store.list_domains()
        for d, pt in domains:
            if d == args.domain:
                structure = await structure_store.get_structure(d, pt)
                break

    if structure is None:
        print(f"No structure found for {args.domain}")
        return

    detector.set_site_baseline(structure.domain, structure)

    # Save detector state
    output = args.detector_state or "detector_state.pkl"
    detector.save(output)

    print(f"Baseline set for {structure.domain} [{structure.page_type}]")
    print(f"Detector state saved to: {output}")


async def cmd_detect_drift(args, structure_store: StructureStore, detector: MLChangeDetector):
    """Detect drift from baseline for a domain."""
    structure = await structure_store.get_structure(args.domain, args.page_type or "homepage")
    if structure is None:
        # Try to find any page type
        domains = await structure_store.list_domains()
        for d, pt in domains:
            if d == args.domain:
                structure = await structure_store.get_structure(d, pt)
                break

    if structure is None:
        print(f"No structure found for {args.domain}")
        return

    result = detector.detect_drift_from_baseline(structure)

    if result is None:
        print(f"No baseline found for {structure.domain}. Run set-baseline first.")
        return

    print(f"\nDrift Analysis for {structure.domain}:")
    print("=" * 60)
    print(f"  Similarity to baseline: {result['similarity_to_baseline']:.2%}")
    print(f"  Is drifted: {result['is_drifted']}")
    print(f"  Baseline created: {result['baseline_created']}")


async def cmd_compare(args, structure_store: StructureStore, detector: MLChangeDetector):
    """Compare two structures using ML change detection."""
    # Get both structures
    old_structure = await structure_store.get_structure(args.domain1, args.page_type or "homepage")
    new_structure = await structure_store.get_structure(args.domain2, args.page_type or "homepage")

    # Try to find structures if not found
    if old_structure is None:
        domains = await structure_store.list_domains()
        for d, pt in domains:
            if d == args.domain1:
                old_structure = await structure_store.get_structure(d, pt)
                break

    if new_structure is None:
        domains = await structure_store.list_domains()
        for d, pt in domains:
            if d == args.domain2:
                new_structure = await structure_store.get_structure(d, pt)
                break

    if old_structure is None:
        print(f"No structure found for {args.domain1}")
        return

    if new_structure is None:
        print(f"No structure found for {args.domain2}")
        return

    result = detector.detect_change(old_structure, new_structure)

    print(f"\nChange Detection: {args.domain1} -> {args.domain2}")
    print("=" * 60)
    print(f"  Embedding similarity: {result['similarity']:.2%}")
    print(f"  Is breaking change: {result['is_breaking']}")

    if 'predicted_impact' in result:
        print(f"  Predicted impact: {result['predicted_impact']} ({result['impact_confidence']:.2%} confidence)")

    print("\nChange Description:")
    print(result['change_description'])


async def main():
    parser = argparse.ArgumentParser(
        description="Train and use ML embeddings with crawled structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL (default: from config)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export training data")
    export_parser.add_argument("--output", "-o", default="training_data.jsonl")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Create embeddings")
    embed_parser.add_argument("--output", "-o", default="embeddings.json")

    # Similar command
    similar_parser = subparsers.add_parser("similar", help="Find similar structures")
    similar_parser.add_argument("domain", help="Domain to find similar to")
    similar_parser.add_argument("--top-k", type=int, default=5)

    # Train classifier command
    train_parser = subparsers.add_parser("train", help="Train classifier")
    train_parser.add_argument("--output", "-o", default="classifier.pkl")
    train_parser.add_argument(
        "--classifier-type",
        choices=["logistic_regression", "xgboost", "lightgbm"],
        default="logistic_regression",
        help="Classifier type (default: logistic_regression)",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict page type")
    predict_parser.add_argument("domain", help="Domain to predict")
    predict_parser.add_argument("--page-type", default=None)
    predict_parser.add_argument("--classifier", default="classifier.pkl")

    # Similarity pairs command
    pairs_parser = subparsers.add_parser("pairs", help="Create similarity pairs")
    pairs_parser.add_argument("--output", "-o", default="similarity_pairs.jsonl")

    # Describe command (rules vs LLM)
    describe_parser = subparsers.add_parser("describe", help="Generate structure description")
    describe_parser.add_argument("domain", help="Domain to describe")
    describe_parser.add_argument("--page-type", default=None)
    describe_parser.add_argument(
        "--mode",
        choices=["rules", "llm"],
        default="rules",
        help="Description mode (default: rules)",
    )
    describe_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama", "ollama-cloud"],
        default="openai",
        help="LLM provider for llm mode (default: openai)",
    )
    describe_parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (default: provider-specific, e.g., gpt-4o-mini, llama3.2)",
    )
    describe_parser.add_argument(
        "--ollama-url",
        default=None,
        help="Custom Ollama base URL (default: http://localhost:11434 for local)",
    )

    # Set baseline command
    baseline_parser = subparsers.add_parser("set-baseline", help="Set baseline for change detection")
    baseline_parser.add_argument("domain", help="Domain to set as baseline")
    baseline_parser.add_argument("--page-type", default=None)
    baseline_parser.add_argument("--detector-state", default="detector_state.pkl")

    # Detect drift command
    drift_parser = subparsers.add_parser("detect-drift", help="Detect drift from baseline")
    drift_parser.add_argument("domain", help="Domain to check for drift")
    drift_parser.add_argument("--page-type", default=None)
    drift_parser.add_argument("--detector-state", default="detector_state.pkl")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two structures")
    compare_parser.add_argument("domain1", help="First domain")
    compare_parser.add_argument("domain2", help="Second domain")
    compare_parser.add_argument("--page-type", default=None)
    compare_parser.add_argument(
        "--mode",
        choices=["rules", "llm"],
        default="rules",
        help="Description mode for change report (default: rules)",
    )
    compare_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama", "ollama-cloud"],
        default="openai",
        help="LLM provider for llm mode (default: openai)",
    )
    compare_parser.add_argument(
        "--ollama-url",
        default=None,
        help="Custom Ollama base URL (default: http://localhost:11434 for local)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Connect to Redis
    try:
        config = load_config()
        redis_url = args.redis_url or config.redis_url
    except Exception:
        redis_url = args.redis_url or "redis://localhost:6379/0"

    client = redis.from_url(redis_url, decode_responses=False)
    structure_store = StructureStore(client)

    # Create ML change detector for relevant commands
    detector = None
    if args.command in ("set-baseline", "detect-drift", "compare"):
        model = StructureEmbeddingModel(model_name=args.model)

        # Get description mode if available
        mode = getattr(args, "mode", "rules")
        if mode == "llm":
            kwargs = {"provider": getattr(args, "provider", "openai")}
            # Add ollama_base_url if specified
            if hasattr(args, "ollama_url") and args.ollama_url:
                kwargs["ollama_base_url"] = args.ollama_url
            desc_gen = get_description_generator(DescriptionMode.LLM, **kwargs)
        else:
            desc_gen = get_description_generator(DescriptionMode.RULES)

        detector = MLChangeDetector(
            embedding_model=model,
            description_generator=desc_gen,
        )

        # Load existing state if available
        if args.command in ("detect-drift", "compare"):
            state_path = getattr(args, "detector_state", "detector_state.pkl")
            try:
                detector.load(state_path)
                print(f"Loaded detector state from {state_path}")
            except FileNotFoundError:
                pass

    try:
        if args.command == "export":
            await cmd_export(args, structure_store)
        elif args.command == "embed":
            await cmd_embed(args, structure_store)
        elif args.command == "similar":
            await cmd_similar(args, structure_store)
        elif args.command == "train":
            await cmd_train_classifier(args, structure_store)
        elif args.command == "predict":
            await cmd_predict(args, structure_store)
        elif args.command == "pairs":
            await cmd_similarity_pairs(args, structure_store)
        elif args.command == "describe":
            await cmd_describe(args, structure_store)
        elif args.command == "set-baseline":
            await cmd_set_baseline(args, structure_store, detector)
        elif args.command == "detect-drift":
            await cmd_detect_drift(args, structure_store, detector)
        elif args.command == "compare":
            await cmd_compare(args, structure_store, detector)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
