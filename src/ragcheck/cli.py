"""Interfaz de línea de comandos: `ragcheck train|evaluate|score`."""

import argparse


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada de la CLI. Enruta al subcomando pedido."""
    parser = argparse.ArgumentParser(
        prog="ragcheck",
        description="Detección de alucinaciones en RAG con ML clásico.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Entrena y persiste el modelo por defecto.")

    p_eval = sub.add_parser("evaluate", help="Evalúa un modelo (secciones A–F).")
    p_eval.add_argument("--model", default="logreg")

    p_score = sub.add_parser("score", help="Puntúa un par (respuesta, fuente).")
    p_score.add_argument("--output", required=True)
    p_score.add_argument("--context", required=True)

    args = parser.parse_args(argv)

    if args.command == "train":
        raise NotImplementedError("train se implementa en la Fase 0")
    if args.command == "evaluate":
        raise NotImplementedError("evaluate se implementa en la Fase 0")
    if args.command == "score":
        from ragcheck.inference import score

        print(score(args.output, args.context))


if __name__ == "__main__":
    main()
