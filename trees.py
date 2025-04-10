import xml.etree.ElementTree as ET
import networkx as nx

class TreeMaker():

    def __init__(self, xml_file) -> None:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        mesh_terms = {}
        for record in root.findall("DescriptorRecord"):
            ui = record.find("DescriptorUI").text
            name = record.find("DescriptorName/String").text
            tree_numbers = [tn.text for tn in record.findall("TreeNumberList/TreeNumber")]
            mesh_terms[name] = {"UI": ui, "TreeNumbers": tree_numbers}

        self.mesh_dict = mesh_terms

    def tree_from_key(self, keyword):
        codes = []
        codes_dict = {}
        for term, val in self.mesh_dict.items():
            tn_list = val['TreeNumbers']
            for num in tn_list:
                if keyword in num:
                    codes.append(num)
                    codes_dict[num] = term
        codes.sort(key=lambda x: x.count('.'))
        code_set = set(codes)
        G = nx.DiGraph()
        for code in codes:
            G.add_node(code)
            if '.' in code:
                parent = code.rsplit('.', 1)[0]
                if parent in code_set:
                    G.add_edge(parent, code)

        return G, codes_dict
        