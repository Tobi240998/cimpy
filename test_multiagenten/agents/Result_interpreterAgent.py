class Result_interpreterAgent:
    """
    Agent responsible for:
    - interpreting results
    - identifying critical voltage levels
    """

    def interpret_voltage_change(self, u_before, u_after):
        results = []  # Erstellen einer leeren Liste für Ergebnisse

        for name in u_before: #Schritte werden für jeden Bus durchgeführt 
            delta = u_after[name] - u_before[name] #Berechnung Spannungsdifferenz
            #Sprachliche Bausteine
            if delta > 0:
                trend = "erhöht"
            elif delta < 0:
                trend = "verringert"
            else:
                trend = "nicht verändert"

            message = (
                f"Bus '{name}': Spannung {trend} um {delta:.4f} V "
                f"(alt: {u_before[name]:.4f} V, neu: {u_after[name]:.4f} V)."
            )
            #Prüfung, ob Über- / Unterspannung vorliegt und ggfs. Ausgabe Hinweis 
            if u_after[name] < 0.95:
                message += " Achtung: Unterspannung (< 0.95 V)."

            if u_after[name] < 0.90:
                message += " Kritische Unterspannung (< 0.90 V)!"

            if u_after[name] > 1.05:
                message += " Warnung: Überspannung (> 1.05 V)."

            results.append(message)

        return results
