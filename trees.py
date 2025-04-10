import xml.etree.ElementTree as ET
import networkx as nx

class TreeMaker():

    def __init__(self, xml_file) -> None:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        names = []
        mesh_terms = {}
        for record in root.findall("DescriptorRecord"):
            ui = record.find("DescriptorUI").text
            name = record.find("DescriptorName/String").text
            tree_numbers = [tn.text for tn in record.findall("TreeNumberList/TreeNumber")]
            mesh_terms[name] = {"UI": ui, "TreeNumbers": tree_numbers}
            names.append(name)
        
        self.names = names
        self.mesh_dict = mesh_terms
    
    def tn_to_name(self, tree_number):
        
        name = ''

        for n, temp in self.mesh_dict.items():
            tn = temp['TreeNumbers']
            if tree_number in tn:

                name = n
                break

        return namex


    def tree_from_key(self, keyword):
        codes = []
        code_names = []
        for term, val in self.mesh_dict.items():
            tn_list = val['TreeNumbers']
            for num in tn_list:
                if keyword in num:
                    codes.append(num)
                    code_names.append(term)

        paired = list(zip(codes, code_names))
        paired.sort(key=lambda x: x[0].count('.'))
        codes, code_names = zip(*paired)
        code_set = set(codes)
        G = nx.DiGraph()
        for code in codes:
            G.add_node(code)
            if '.' in code:
                parent = code.rsplit('.', 1)[0]
                if parent in code_set:
                    G.add_edge(parent, code)

        return G, code_names
    

        