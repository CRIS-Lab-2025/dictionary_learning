import random
from typing import Dict, List
import xml.etree.ElementTree as ET
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree


class Record():
    def __init__(self, name, ui, tree_numbers, note, tree):
        self.name:str = name
        self.ui:str = ui
        self.tree_numbers:List[str] = tree_numbers
        self.note:str = note
        
        self.tree:nx.DiGraph = tree # Tree object
        
    def get_tree_numbers(self):
        return self.tree_numbers
    
    def add_tree_number(self, tree_number):
        if tree_number not in self.tree_numbers:
            self.tree_numbers.append(tree_number)
            
    def get_id(self):
        return self.ui
    
    def get_name(self):
        return self.name
    
    def get_note(self):
        return self.note
    
    def get_ancestors(self,group_by_depth=False):
        

        anc = nx.ancestors(self.tree, self)
        
        if group_by_depth:
            
            groups = []
            
            subtree = self.tree.subgraph(anc|{self}).reverse()
            
            for d in anc:
                depth = nx.shortest_path_length(subtree, source=self, target=d)
                
                if len(groups) < depth:
                    for _ in range(depth - len(groups)):
                        groups.append([])
                
                groups[depth-1].append(d)
                
            return groups
        
        return list(anc)
    
    def get_parents(self):
        pred = self.tree.predecessors(self)
        
        return list(pred)
    
    def get_children(self):
        succ = self.tree.successors(self)
        
        return list(succ)
    
    def get_descendants(self,group_by_depth=False):
        

        desc = nx.descendants(self.tree, self)
        
        if group_by_depth:
            
            groups = []
            
            subtree = self.tree.subgraph(desc|{self})
            
            for d in desc:
                depth = nx.shortest_path_length(subtree, source=self, target=d)
                
                if len(groups) < depth:
                    for _ in range(depth - len(groups)):
                        groups.append([])
                
                groups[depth-1].append(d)
                
            return groups
        
        return list(desc)
    
    
    def get_subtree(self, max_depth=None):
        """
        Gets a subtree of max_depth with the current node as root.
        """
        
        return dfs_tree(self.tree, self, depth_limit=max_depth)
        
        
    
    
    def __str__(self):
        return f"Record(name={self.name}, ui={self.ui})"

    def __repr__(self):
        return f"Record(name={self.name}, ui={self.ui}, tree_numbers={self.tree_numbers}, note={self.note} tree={self.tree})"

class Tree():



    def __init__(self, xml_file) -> None:
        self.mesh_dict:Dict[Record] = {}
        self.graph = nx.DiGraph()
        self.name_to_id = {}
        self.tn_to_id = {}
        
        
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for record in root.findall("DescriptorRecord"):
            
            
            ui = record.find("DescriptorUI").text
            name = record.find("DescriptorName/String").text
            name_lower = name.lower()
            tree_numbers = [tn.text for tn in record.findall("TreeNumberList/TreeNumber")]
            note = record.find("ConceptList/Concept/ScopeNote").text if record.find("ConceptList/Concept/ScopeNote") is not None else None
            
            record = Record(name, ui, tree_numbers, note, self.graph)
            
            self.mesh_dict[ui] = record
            self.name_to_id[name_lower] = ui
            self.tn_to_id |= {tn: ui for tn in tree_numbers}
            self.graph.add_node(record)
            
        for record in self.mesh_dict.values():
            for tn in record.get_tree_numbers():
                parent = tn.rsplit('.', 1)[0]
                
                parent_ui = self.tn_to_id.get(parent, None)
                
                if parent_ui and parent_ui != record.get_id():
                    parent_record = self.mesh_dict[parent_ui]
                    self.graph.add_edge(parent_record, record)
            
            
            


    
    def get_node(self, identifier:str):
        identifier = identifier.lower()

        by_name = self.name_to_id.get(identifier, None)
        if by_name:
            record = self.mesh_dict.get(by_name)
            return record
        
        by_tn = self.tn_to_id.get(identifier, None)
        if by_tn:
            record = self.mesh_dict.get(by_tn)
            return record
        
        record = self.mesh_dict.get(identifier, None)
        if record:
            return record
        
        
        return None
    
    def sample_node(self):
        
        return random.choice(list(self.graph.nodes()))
    
        
    

        