from cimpy.powerfactory_agent.pf_runner import _import_powerfactory, _get_app
from cimpy.powerfactory_agent.powerfactory_mcp_tools import get_load_catalog

pf = _import_powerfactory()
app = _get_app(pf)

print("PowerFactory App verbunden:", app)

project = app.GetActiveProject()
studycase = app.GetActiveStudyCase()

print("Aktives Projekt:", project)
print("Aktiver StudyCase:", studycase)

result = get_load_catalog(project, studycase)
print(result["status"])
print(result["loads"][:3])