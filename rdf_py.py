import rdflib

graph = rdflib.Graph("Sleepycat")
graph.open("store", create=True)
graph.parse("aact_ali01.rdf")

# print out all the triples in the graph
for subject, predicate, object in graph:
    print (subject, predicate, object)