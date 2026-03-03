#Werkzeuge importieren
import ollama #Sprachmodell
import json #JSON-Format
import re #Funktion für Textmuster
import sys
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP7\Python\3.10") #Pfad für PowerFactory
import powerfactory as pf #PowerFactory
#Agenten importieren
from agents.PowerFactoryAgent import PowerFactoryAgent 
from agents.LLM_interpreterAgent import LLM_interpreterAgent 
from agents.Result_interpreterAgent import Result_interpreterAgent
from agents.LLM_resultAgent import LLM_resultAgent


#PowerFactory starten
app = pf.GetApplication()
if app is None:
    raise RuntimeError("PowerFactory nicht erreichbar")

#Projekt aktivieren 
project_name = "Nine-bus System(2)"
app.ActivateProject(project_name)

#Aktives Projekt holen
project = app.GetActiveProject()
if project is None:
    raise RuntimeError("Projekt nicht aktiv")

#Aktiven Studycase holen 
studycase = app.GetActiveStudyCase()
if studycase is None:
    raise RuntimeError("Kein aktiver Berechnungsfall")

#Ausgabe des aktiven Studycases
print("Study Case:", studycase.loc_name)

ldf_list = studycase.GetContents("*.ComLdf") #Ausgabe aller Lastflussberechnungen im Studycase

if not ldf_list:
    ldf = studycase.CreateObject("ComLdf", "LoadFlow") #falls kein Lastfluss existiert -> einen erstellen
else:
    ldf = ldf_list[0] #falls Lastfluss vorhanden, den ersten nehmen

#Agenten instanziieren
pf_agent = PowerFactoryAgent(project, studycase)
llm_agent = LLM_interpreterAgent(project, studycase)
result_agent = Result_interpreterAgent()
llm_result_agent = LLM_resultAgent()

#LLM-Eingabe
user_input = input("Anweisung: ")

#Interpretation der Eingabe
instruction = llm_agent.interpret(user_input)
print("LLM instruction:", instruction)

#Last identifizieren
resolved_load = llm_agent.resolve(instruction)


#Lastfluss vor der Änderung

ldf.Execute() 



#Spannungen vor der Änderung speichern
buses = app.GetCalcRelevantObjects("*.ElmTerm")

u_before = {}
for bus in buses:
    name = bus.loc_name
    u_before[name] = bus.GetAttribute("m:u")


#Last ändern
try:
    p_old = resolved_load.GetAttribute("plini")
except AttributeError:
    raise RuntimeError(f"Last {resolved_load.loc_name} hat kein Attribut 'plini'")

pf_agent.execute(instruction, resolved_load)

#Lastfluss nach der Änderung

ldf.Execute()


#Neue Spannungswerte speichern
u_after = {}
for bus in buses:
    name = bus.loc_name
    u_after[name] = bus.GetAttribute("m:u")


#Vergleich berechnen und ausgeben

print("\n--- Spannungsänderungen ---")
for name, u0 in u_before.items():
    u1 = u_after.get(name)
    if u1 is None:
        print(f"{name:20s}: kein Ergebnis nach der Änderung")
        continue

    delta = u1 - u0
    print(f"{name:20s}: {delta:+.5f} V")

#Faktische Interpretation der Ergebnisse 

messages = result_agent.interpret_voltage_change(u_before, u_after)
print(messages)

#Interpretation der Ergebnisse durch das LLM auf Basis der faktischen Interpretation

summary = llm_result_agent.summarize(messages, user_input)

print("\n--- LLM-Zusammenfassung ---") 
print(summary)
