class Result_interpreterAgent:
    """
    Agent responsible for:
    - interpreting results
    - identifying critical voltage levels
    """

    def interpret_voltage_change(self, u_before, u_after, voltage_limits=None):
        results = []  # Erstellen einer leeren Liste für Ergebnisse
        buses_with_violation = []
        buses_within_limits = []
        buses_without_limits = []

        for name in u_before: #Schritte werden für jeden Bus durchgeführt
            if name not in u_after:
                continue

            delta = u_after[name] - u_before[name] #Berechnung Spannungsdifferenz
            #Sprachliche Bausteine
            if delta > 0:
                trend = "erhöht"
            elif delta < 0:
                trend = "verringert"
            else:
                trend = "nicht verändert"

            message = (
                f"Bus '{name}': Spannung {trend} um {delta:.4f} p.u. "
                f"(alt: {u_before[name]:.4f} p.u., neu: {u_after[name]:.4f} p.u.)."
            )

            umin = None
            umax = None
            if isinstance(voltage_limits, dict):
                limits = voltage_limits.get(name, {})
                if isinstance(limits, dict):
                    umin = limits.get("umin")
                    umax = limits.get("umax")

            if umin is not None or umax is not None:
                limit_parts = []
                violated = False

                if umin is not None:
                    limit_parts.append(f"Umin={umin:.4f} p.u.")
                if umax is not None:
                    limit_parts.append(f"Umax={umax:.4f} p.u.")

                if limit_parts:
                    message += " Grenzwerte: " + ", ".join(limit_parts) + "."

                if umin is not None and u_after[name] < umin:
                    message += f" Achtung: Unterspannung (< {umin:.4f} p.u.)."
                    violated = True

                if umax is not None and u_after[name] > umax:
                    message += f" Warnung: Überspannung (> {umax:.4f} p.u.)."
                    violated = True

                if violated:
                    buses_with_violation.append(name)
                else:
                    message += " Spannung innerhalb der in PowerFactory hinterlegten Grenzen."
                    buses_within_limits.append(name)
            else:
                # Fallback auf bisherige Standardgrenzen, falls keine PF-Grenzen verfügbar sind
                fallback_violation = False

                if u_after[name] < 0.95:
                    message += " Achtung: Unterspannung (< 0.95 p.u.)."
                    fallback_violation = True

                if u_after[name] < 0.90:
                    message += " Kritische Unterspannung (< 0.90 p.u.)!"
                    fallback_violation = True

                if u_after[name] > 1.05:
                    message += " Warnung: Überspannung (> 1.05 p.u.)."
                    fallback_violation = True

                if not fallback_violation:
                    message += " Spannung innerhalb der Standardgrenzen (Fallback: 0.95 bis 1.05 p.u.)."
                    buses_within_limits.append(name)
                else:
                    buses_with_violation.append(name)

                buses_without_limits.append(name)

            results.append(message)

        if buses_with_violation:
            results.append(
                "Grenzwertverletzungen wurden festgestellt bei: "
                + ", ".join(sorted(buses_with_violation))
                + "."
            )
        elif buses_within_limits:
            results.append(
                "Alle betrachteten Busspannungen liegen innerhalb der zulässigen Spannungsgrenzen."
            )

        if buses_without_limits:
            results.append(
                "Hinweis: Für folgende Busse wurden keine PowerFactory-Grenzwerte gefunden; es wurden Standardgrenzen verwendet: "
                + ", ".join(sorted(set(buses_without_limits)))
                + "."
            )

        return results
