from pathlib import Path
from typing import Any


def _ratio(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})" if total else "0/0 (n/a)"


def render_summary(report: dict[str, Any], output: Path) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    results = report["results"]
    total = len(results)
    evidence = summary["required_evidence_coverage"]
    facts = summary["factual_coverage"]
    attribution = summary["evidence_attribution"]
    format_counts = summary["format_compliance"]
    schema_counts = summary["response_format_status"]
    lines = [
        f"Evaluación {metadata['mode'].upper()} | {metadata['strategy']} | "
        f"{metadata['output_mode']}",
        f"Escenarios: {summary['scenarios']} | Intentos por escenario: {metadata['trials']} | "
        f"Ejecuciones: {total}",
    ]
    if metadata["mode"] == "live":
        model = metadata["model"] or "desconocido"
        model_name = model.replace("\\", "/").rsplit("/", 1)[-1]
        lines.append(f"API: {metadata['api_base_url']} | Modelo: {model_name}")
    lines.extend(
        [
            "",
            "Resultados",
            f"  Éxito: {_ratio(summary['task_successes'], total)} | "
            f"Completadas: {_ratio(summary['completed'], total)} | "
            f"Fallos de infraestructura: {summary['infrastructure_errors']}",
            f"  Selección de herramientas: {_ratio(summary['tool_selection_passed'], total)} | "
            f"Fallos de aserciones: {summary['evaluation_failures']}",
            "Cobertura",
            f"  Evidencia obtenida: {_ratio(evidence['covered'], evidence['total'])} "
            f"({evidence['attempted']} intentos)",
            f"  Frases requeridas presentes: {_ratio(facts['covered'], facts['total'])} | "
            f"Fuentes atribuidas: {_ratio(attribution['matched'], attribution['total'])}",
            f"  Coincidencias de afirmaciones sin sustento: "
            f"{summary['approximate_unsupported_claim_screen']['matches']} "
            "(detección aproximada)",
            "Formato",
            f"  Válido: {format_counts['valid']} | Inválido: {format_counts['invalid']} | "
            f"No producido: {format_counts['not_produced']} | No aplica: "
            f"{format_counts['not_applicable']}",
            f"  Esquema aceptado: {schema_counts['accepted']} | Rechazado: "
            f"{schema_counts['rejected']} | Desconocido: {schema_counts['unknown']} | "
            f"No solicitado: {schema_counts['not_requested']}",
            "Rendimiento",
            f"  Latencia media: {summary['mean_latency_ms']:.1f} ms | "
            f"Mediana: {summary['median_latency_ms']:.1f} ms",
            f"  Herramientas por ejecución: {summary['mean_tool_calls']:.1f} | "
            f"Errores de herramienta: {summary['mean_tool_errors']:.1f}",
            f"  Tokens medios: {summary['mean_prompt_tokens']:.1f} entrada + "
            f"{summary['mean_completion_tokens']:.1f} salida",
            "",
            "Por escenario (éxito / selección / aserciones / infraestructura | latencia media)",
        ]
    )
    if metadata["mode"] == "live":
        runtime_mean = summary["mean_runtime_latency_ms"]
        lines.insert(
            lines.index("Rendimiento") + 2,
            "  Ejecución media del servidor: "
            + (f"{runtime_mean:.1f} ms" if runtime_mean is not None else "n/a"),
        )
    for test_id, row in summary["by_scenario"].items():
        attempts = row["trials"]
        assertions = attempts - row["evaluation_failures"]
        lines.append(
            f"  {test_id:<24} {row['task_successes']}/{attempts} | "
            f"{row['tool_selection_passed']}/{attempts} | {assertions}/{attempts} | "
            f"{row['infrastructure_errors']}/{attempts} | {row['mean_latency_ms']:.1f} ms"
        )
    failures = [row for row in results if not row["task_success"]]
    if failures:
        lines.extend(["", "Fallos destacados"])
        for row in failures[:10]:
            if row["termination_reason"].startswith("api_transport_"):
                causes = [
                    "conexión API " + row["termination_reason"].removeprefix("api_transport_")
                ]
            elif row["infrastructure_error"] and row["api_status_code"]:
                causes = [f"API HTTP {row['api_status_code']}"]
            else:
                causes = []
                if not row["tool_selection_accuracy"]:
                    causes.append("selección de herramientas")
                missing_facts = [
                    fact for fact, state in row["required_facts"].items() if state != "covered"
                ]
                if missing_facts:
                    causes.append("frases faltantes: " + ", ".join(missing_facts))
                if row["output_schema"] != "compliant":
                    causes.append("esquema de respuesta")
                if "invalid" in row["tool_argument_validity"]:
                    causes.append("argumentos de herramienta")
                if row["execution_errors"] != row["expected_tool_errors"]:
                    causes.append("errores de herramienta inesperados")
                if row["status"] != "completed":
                    causes.append(f"estado {row['status']}")
                if not causes:
                    causes.append("criterios de evaluación")
            lines.append(f"  {row['test_id']} intento {row['trial']}: causas: " + "; ".join(causes))
        if len(failures) > 10:
            lines.append(f"  ... y {len(failures) - 10} más en el reporte JSON")
    if comparison := report.get("baseline_comparison"):
        lines.extend(
            [
                "",
                "Cambio frente al reporte base",
                f"  Éxitos: {comparison['task_success_delta']:+d} | "
                f"Selección de herramientas: {comparison['tool_selection_delta']:+d} | "
                f"Completadas: {comparison['completed_delta']:+d}",
            ]
        )
    lines.extend(["", f"Reporte JSON completo: {output}"])
    return "\n".join(lines)
