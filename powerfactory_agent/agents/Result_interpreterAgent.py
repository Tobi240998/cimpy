class Result_interpreterAgent:
    """
    Agent responsible for:
    - interpreting results
    - identifying critical voltage levels
    """

    def interpret_voltage_change(self, u_before, u_after, voltage_limits=None):
        results = []  # Erstellen einer leeren Liste für Ergebnisse

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
            #Prüfung, ob Über- / Unterspannung vorliegt und ggfs. Ausgabe Hinweis 
            limits_for_bus = (voltage_limits or {}).get(name, {}) if isinstance(voltage_limits, dict) else {}
            umin = limits_for_bus.get("umin")
            umax = limits_for_bus.get("umax")

            if umin is not None and u_after[name] < umin:
                message += f" Achtung: Unterspannung (< {umin:.4f} p.u.)."

            if umin is not None and u_after[name] < (umin - 0.05):
                message += f" Kritische Unterspannung (< {umin - 0.05:.4f} p.u.)!"

            if umax is not None and u_after[name] > umax:
                message += f" Warnung: Überspannung (> {umax:.4f} p.u.)."

            # Fallback auf bisherige Standardgrenzen, wenn keine PF-Grenzen vorhanden sind
            if umin is None and umax is None:
                if u_after[name] < 0.95:
                    message += " Achtung: Unterspannung (< 0.95 p.u.)."

                if u_after[name] < 0.90:
                    message += " Kritische Unterspannung (< 0.90 p.u.)!"

                if u_after[name] > 1.05:
                    message += " Warnung: Überspannung (> 1.05 p.u.)."

            results.append(message)

        return results
