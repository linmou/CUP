import json
import queue
from typing import List
import re
from collections import defaultdict
import argparse
import torch
import ipdb
import penman
from penman.graph import Graph
from penman.codec import PENMANCodec
from penman import layout
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging
# from transition_amr_parser.parse import AMRParser


coref_info={"document": ["2014", ":", "ISIS", "marches", "into", "Raqqa", "and", "makes", "it", "the", "capital", "of", "the", "caliphate", ".", "Foto", ":", "Uncredited", "The", "two", "Syrian", "women", "who", "we", "shall", "call", "Om", "Omran", "and", "Om", "Mohammad", ",", "which", "are", "not", "their", "real", "names", ",", "were", "willing", "to", "wear", "Expressen", "'s", "hidden", "cameras", ",", "which", "have", "been", "smuggled", "in", ".", "\"", "We", "want", "the", "world", "to", "know", ",", "\"", "they", "say", ".", "Over", "the", "course", "of", "several", "weeks", ",", "they", "have", "documented", "life", "in", "the", "completely", "isolated", "city", "of", "al", "-", "Raqqah", "."],
            "clusters": [[[5, 5], [8, 8], [78, 85]], [[18, 37], [35, 35], [55, 55], [63, 63], [73, 73]]]}

class sentAmrGraph():
    '''
    load an Amr graph for a single sentence, with edge information
    '''
    def __init__(self, raw_nodes, raw_edges, tokens):
        ipdb.set_trace()
        self.node_id2idx, self.abandon_wordidx = self.process_nodes(raw_nodes)  # {id:(st,ed)}
        self.edges = self.process_edges(raw_edges)  # {(st,ed):[(type,(st,ed)),....]}
        self.tokens = tokens
        self.process_graph(re.compile(r'^op\d+'))
        self.process_graph(re.compile(r'name'))

    def process_nodes(self, nodes):
        node_id2idx = {}
        abandon_wordidx = []
        proper_noun = re.compile(r'"(\S+)"')
        for node in nodes:
            node_id2idx[node[0]] = (int(node[2]), int(node[3]))
            if node[1] in ['and', 'or', 'not'] or (
                    proper_noun.match(node[1]) is not None):  # Remove concat words and proper nouns from pretraining
                abandon_wordidx.append((int(node[2]), int(node[3])))
        return node_id2idx, abandon_wordidx

    def process_edges(self, raw_edges):
        edges = defaultdict(list)
        for raw_edge in raw_edges:
            node1, node2 = raw_edge[3], raw_edge[4]
            node1idx = self.node_id2idx[node1] if node1 in self.node_id2idx else node1
            node2idx = self.node_id2idx[node2] if node2 in self.node_id2idx else node2
            if node1idx == node2idx:  # remove self circle. This will also remove edges like:   country ---:name---> name ---:op---> U.S. if they are same word
                continue
            if raw_edge[1].endswith('-of'):  # reverse edge
                edges[node2idx].append((raw_edge[1][:-3], node1idx))
            else:
                edges[node1idx].append((raw_edge[1], node2idx))
        return edges

    def process_graph(self, rule):
        edges = self.edges
        edges_tuple = [(k, v[0], v[1]) for k, vs in edges.items() for v in vs]
        new_op_edges_tuple = []
        for edge_tuple in edges_tuple:
            node1idx = edge_tuple[0]
            edge_rel = edge_tuple[1]
            node2idx = edge_tuple[2]
            if rule.match(edge_rel) is not None:  # Merge all op/name nodes to its parent nodes
                new_edges_tuple = [(e_t[0], e_t[1], node2idx) for e_t in edges_tuple if e_t[2] == node1idx]
                new_op_edges_tuple.extend(new_edges_tuple)

        for new_edges_tuple in new_op_edges_tuple:
            n1, rel, n2 = new_edges_tuple
            self.edges[n1].append((rel, n2))

class DocumentalAmrGraph():
    def __init__(self,raw_amrs):
        self.tokens, self.abandon_wordidx=[],[]
        self.edges=defaultdict(list)
        self.node_id2idx,self.concept2idx,self.coref_info={},{},{}
        self.spanized_amr,self.raw_amr,self.PENMANgraphs,self.docu_tks=[],[],[],[]
        token_begin_num=0
        for amrid,(nodes, edges, tokens,amrlines,short_alignments) in enumerate(raw_amrs):
            amrid=str(amrid)
            tokens.remove('<ROOT>')
            nodes=[(amrid+'.'+node[0],node[1],token_begin_num+int(node[2]),token_begin_num+int(node[3])) for node in nodes]
            edges=[ (edge[0],edge[1],edge[2],amrid+'.'+edge[3],amrid+'.'+edge[4]) for edge in edges]
            token_begin_num+=len(tokens)
            self.tokens.append(tokens)
            self.docu_tks+=tokens
            node_id2idx, abandon_wordidx= self.process_nodes(nodes)
            self.node_id2idx.update(node_id2idx)
            self.abandon_wordidx.extend(abandon_wordidx)
            self.edges.update(self.process_edges(node_id2idx,edges))
            self.raw_amr.append(amrlines)
            # self.spanized_amr.append(self.process_amr(amrid, nodes,amrlines,short_alignments))
            self.concept2idx.update(self.reverse_align(short_alignments,amrid,node_id2idx))
            # self.PENMANgraphs.append(self.PENMANtriples(amrlines,short_alignments,amrid))

        self.merge_same_nodes()

    def reverse_align(self,short_alignments,amrid,node_id2idx):
        if short_alignments=={}: return {}
        doc_nodeid = [amrid + '.' + nodeid for nodeid in list(short_alignments.keys())]
        doc_tkid=[node_id2idx[nid] if nid in node_id2idx else nid for nid in doc_nodeid]
        doc_value = [val+ '.' + amrid for val in list(short_alignments.values())]
        reversed_align = dict(zip(doc_value, doc_tkid))
        return reversed_align


    def merge_same_nodes(self):
        clusters=defaultdict(list)
        for node,span in self.node_id2idx.items():
            key=' '.join(self.docu_tks[span[0]:span[1]]).lower()
            if span not in clusters[key]:
                clusters[key].append(span)
        for sm_id,cluster in enumerate(clusters.values()):
            if len(cluster)>1:
                for span in cluster:
                    self.edges[(f's{sm_id}', 's')].append(('same', tuple(span)))  # for consistency, transform expression of same node to tuple
                    self.edges[tuple(span)].append(('-same', (f's{sm_id}', 's')))

    def merge_coref(self,coref_info):
        coref_info['document']=(' '.join(coref_info['document']).replace('—','-')).split(' ')
        # align spans in coref_info with self.tokens
        # if sum(self.tokens,[])!=coref_info['document']:
        coref2amr=self.align_tokens(sum(self.tokens,[]),coref_info['document'])

        repres=[] # represent text for each cluster
        for cluster in coref_info['clusters']:
            repre=''
            for span in cluster:
                # for each cluster, find the representative text
                sub_span=self.find_substitute((span[0],span[1]+1))
                if sub_span!=None:
                    instance=' '.join(coref_info['document'][sub_span[0]:sub_span[1]])
                else:
                    instance=' '.join(coref_info['document'][span[0]:span[1]+1])

                if len(instance)>len(repre):
                    repre=instance

                span = (coref2amr[span[0]],coref2amr[span[1]])

            repres.append(repre)

        self.coref_info=dict(zip(repres,coref_info['clusters']))

        for cls_id,cluster in enumerate(coref_info['clusters']):
            for span in cluster:
                span[1]+=1
                span=tuple(span)
                if span not in self.node_id2idx.values(): # since the span segments from coref parser differ from those from amr parser, we need to find the corresponse substitute
                    span=self.find_substitute(span)
                    if span==None:continue

                self.edges[(f'c{cls_id}','c')].append(('coreference',tuple(span))) # for consistency, transform exoression of coreference node to tuple
                self.edges[tuple(span)].append(('-coreference',(f'c{cls_id}','c')))

    def find_substitute(self,span):
        MaxIntersection = 0
        substitute = None
        span=tuple(span)
        for amrspan in list(self.node_id2idx.values()):
            if (MaxIntersection < self.get_intersection(amrspan, span)):
                MaxIntersection = self.get_intersection(amrspan, span)
                substitute = amrspan

            elif substitute!=None and MaxIntersection == self.get_intersection(amrspan, span)\
                and len(''.join(self.docu_tks[substitute[0]:substitute[1]]))< len(''.join(self.docu_tks[amrspan[0]:amrspan[1]])):
                substitute = amrspan
        return substitute

    def get_intersection(self,a,b):
        return len(set(range(a[0],a[1])) & set(range(b[0],b[1])))

    def process_amr(self,amrid,nodes, amrlines:List, short_alignments):
        """
        replace node names by tokens in sentences
        """
        if short_alignments=={}: return amrlines
        doc_nodeid = [amrid + '.' + nodeid for nodeid in list(short_alignments.keys())]
        reversed_align = dict(zip(list(short_alignments.values()), doc_nodeid))

        new_amrlines=amrlines.copy()
        quoted_name=re.compile(r'\(([a-z]+\d*)\s')
        unquoted_name=re.compile(r'\s(\w+)')
        for lineid,line in enumerate(new_amrlines):
            name=re.findall(quoted_name,line)
            if len(name)==0:
                name = re.findall(unquoted_name, line)
            if len(name) == 0:
                continue
            for nm in name:
                if nm in reversed_align:
                    try:
                        tok_span=self.node_id2idx[reversed_align[nm]]
                    except:
                        continue
                        # ipdb.set_trace()

                else: # numbers , find them in nodes
                    for nd in nodes:
                        if nd[1]==nm:
                            tok_span=self.node_id2idx[nd[0]]
                            break
                try:
                    new_amrlines[lineid] = line.replace(' ' + nm, ' ' + ','.join(map(str, tok_span)))
                    new_amrlines[lineid] = new_amrlines[lineid].replace('(' + nm, '(' + ','.join(map(str, tok_span)))
                except: ipdb.set_trace()

        return new_amrlines

    def PENMANtriples(self, amr_str:str, short_alignments:dict, amrid:int):
        '''
        return a list of triples (concept.amrid ,rel, concept.amrid) (concept.amrid ,:instance, text_in_doc)
         indicating an amr graph for a single sentence
        '''
        codec = PENMANCodec()
        graph = codec.decode(amr_str)
        nodes2edges = defaultdict(list)
        triples = []

        for tid, trp in enumerate(graph.triples):
            concept = trp[0]+f'.{amrid}' # add amrid information
            repre=trp[2]+f'.{amrid}' if trp[2]!='-' else '-'
            if trp[1] == ':instance':
                try:
                    tok_span = self.concept2idx[concept]
                except:
                    continue

                unique = True
                for corfrepre, cluster in self.coref_info.items():
                    if tok_span in cluster:
                        unique = False
                        concept= 'cf' + corfrepre[:2]
                        repre=corfrepre
                        break

                if unique:
                    try:
                        repre=' '.join(self.docu_tks[tok_span[0]:tok_span[1]])
                    except:
                        pass

            triples.append((concept, trp[1], repre))

        return triples

    def align_tokens(self,amr_parsed:List[str],coref_parsed:List[str]):
        """
        align tokens from amr parser and coreferencer
        Hope that the entities which consist the nodes would not be splited diffferently
        """
        if not len(amr_parsed)<= len(coref_parsed):
            ipdb.set_trace()
        coref2amr={}
        i,j=0,0
        while i < len(amr_parsed):
            coref2amr[j] = i
            if amr_parsed[i] != coref_parsed[j]:
                origin_j=j
                if not ((len(amr_parsed[i]) > len(coref_parsed[j]) or amr_parsed[i].startswith(coref_parsed[j]))):
                    ipdb.set_trace()
                while amr_parsed[i] != ''.join(coref_parsed[origin_j:j+1]) and j<len(coref_parsed):
                    j+=1
                    coref2amr[j] = i

                if amr_parsed[i] != ''.join(coref_parsed[origin_j:j+1]):
                    ipdb.set_trace()

            i+=1
            j+=1

        return coref2amr


    def process_nodes(self, nodes):
        node_id2idx = {}
        abandon_wordidx = []
        proper_noun = re.compile(r'"(\S+)"')
        for node in nodes:
            node_id2idx[node[0]] = (node[2], node[3])
            if node[1] in ['and', 'or', 'not'] or (
                    proper_noun.match(node[1]) is not None):  # Remove concat words and proper nouns from pretraining
                abandon_wordidx.append((node[2], node[3]))
        return node_id2idx, abandon_wordidx

    def process_edges(self,node_id2idx, raw_edges):
        edges = defaultdict(list)
        for raw_edge in raw_edges:
            node1, node2 = raw_edge[3], raw_edge[4]
            node1idx = node_id2idx[node1] if node1 in node_id2idx else node1
            node2idx = node_id2idx[node2] if node2 in node_id2idx else node2
            if node1idx == node2idx:  # remove self circle. This will also remove edges like:   country ---:name---> name ---:op---> U.S. if they are same word
                continue
            if raw_edge[1].endswith('-of'):  # reverse edge
                edges[node2idx].append((raw_edge[1][:-3], node1idx))
                edges[node1idx].append(('-'+raw_edge[1][:-3], node2idx))
            else:
                edges[node1idx].append((raw_edge[1], node2idx))
                edges[node2idx].append(('-'+raw_edge[1], node1idx))

        self.process_graph(edges,re.compile(r'^op\d+'))
        self.process_graph(edges,re.compile(r'name'))

        return edges

    def process_graph(self,edges ,rule):
        edges_tuple = [(k, v[0], v[1]) for k, vs in edges.items() for v in vs]
        new_op_edges_tuple = []
        for edge_tuple in edges_tuple:
            node1idx = edge_tuple[0]
            edge_rel = edge_tuple[1]
            node2idx = edge_tuple[2]
            if rule.match(edge_rel) is not None:  # Merge all op/name nodes to its parent nodes
                new_edges_tuple = [(e_t[0], e_t[1], node2idx) for e_t in edges_tuple if e_t[2] == node1idx and '-' not in e_t[1]]
                new_op_edges_tuple.extend(new_edges_tuple)

        for new_edges_tuple in new_op_edges_tuple:
            n1, rel, n2 = new_edges_tuple
            assert '-' not in rel
            edges[n1].append((rel, n2))
            edges[n2].append(('-'+rel, n1))

    def BFS(self,start_span,target_span):
        q=queue.Queue()
        start_span=self.find_substitute(start_span)
        target_span=self.find_substitute(target_span)
        if None in [start_span,target_span]:
            return

        if start_span==target_span:
            return [start_span]

        Found,stop_loop = False,False
        path = defaultdict(list)
        q.put((-1,start_span,0))
        step_num=1
        visited=set()
        visited.add(start_span)
        while (not q.empty()) and stop_loop==False:
            # ipdb.set_trace()
            (start,cur,pathDir)=q.get()
            if (start,cur,pathDir)==(-1,-1,0):
                step_num+=1
                if Found:
                    stop_loop=True
                continue

            visited.add(cur)
            path[cur].append((start,pathDir))
            if cur == target_span:
                Found = True

            for rel, end_node in self.edges[tuple(cur)]:
                if end_node not in visited:
                    q.put((cur,end_node,rel)) # record the start, end and the dirction of the path

            q.put((-1,-1,0))
        if q.empty() or step_num > len(self.docu_tks): return

        # load found route
        routes=[]
        self.DFS(target_span, start_span, path, [(target_span,1)], routes, step_num)
        shortest_rt= min((len(rt) for rt in routes))
        routes=set( tuple(rt) for rt in routes if len(rt)==shortest_rt)
        parallel_path=[]
        for s in range(shortest_rt):
            parallel_path.append(set(rt[s] for rt in routes))

        return list((parallel_path))


    def DFS(self,start_span,target_span,graph,single_route,routes,step):
        if step<0: return
        if start_span==target_span:
            routes.append(single_route.copy())
            return

        for node,pathDir in graph[start_span]:
            single_route.append((node,pathDir))
            self.DFS(node, target_span, graph,  single_route, routes,step-1)
            single_route.pop()

    def subgraph(self,path,type:str='penman'):
        """
        get the subgraph that contains the path found by BFS
        """
        all_spans,all_edges=set(),set()
        for nid,node in enumerate(path):
            # TODO: consider parallel paths
            n=list(node)[0] #if len(list(node))==1 else list(node)[1]
            if n[0][1]=='c' or n[0][1]=='s':continue
            all_spans.add(n[0])
            if nid!=0: # organized like a piece of shit ....
                previous_nd=list(path[nid-1])
                if n[1]!=1 and n[1][0]=='-':
                    all_edges.add((previous_nd[0][0],':'+n[1][1:],n[0]))
                else:
                    all_edges.add((n[0],':'+n[1],previous_nd[0][0]))


        if type=='penman':
            #TODO: intersentence samples
            final_triples=[]
            graph=self.PENMANgraphs[2]
            # ipdb.set_trace()
            for trp in graph:
                if self.concept2idx[trp[0]] in all_spans and trp[1]==':instance':
                    final_triples.append(trp)
                    # continue
                elif trp[1]!=':instance':
                    if trp[2]=='-' and self.concept2idx[trp[0]] in all_spans and trp[1]==':polarity':
                        # final_triples.append(trp)
                        continue
                    elif trp[1]==':polarity':
                        continue

                    else:
                        try:
                            span_trp=(self.concept2idx[trp[0]],trp[1],self.concept2idx[trp[2]])
                        except:
                            continue
                        if span_trp in all_edges:
                            final_triples.append(trp)


            print(final_triples)
            return PENMANCodec().encode(Graph(final_triples))

        spanized_subgraph, raw_subgraph= [], []
        for spanized_line,raw_line in zip(sum(self.spanized_amr, []), sum(self.raw_amr, [])):
            span=re.findall(r'\d+,\d+', spanized_line)
            if span==[]: continue
            raw_subgraph.append(raw_line)
            span=tuple(map(int,span[0].split(',')))
            if span in all_spans:
                text=' '.join(self.docu_tks[span[0]:span[1]])

                # use structured graph
                if type=='structured':
                    line=re.sub(r'\d+,\d+',text,spanized_line)
                    line=re.sub(r'([a-zA-Z]+)\d',r'\1',line)
                    spanized_subgraph.append(line)
                elif type=='natural':
                    # use natural language spans only
                    spanized_subgraph.append(text)

                all_spans.remove(span)
        if not len(all_spans)==0:
            return None
            # ipdb.set_trace()
        return ' '.join(spanized_subgraph)


def load_document_amr(filename):
    with open(filename, 'r',encoding='utf8') as f:
        amrs = f.read()

    amrs = amrs.split('\n\n')
    amrs = amrs[:-1] if amrs[-1] == "" else amrs
    node_format = re.compile(r'# ::node\t(\S+)\t(\S+)\t(\d+)-(\d+)')
    edge_format = re.compile(r'# ::edge\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)')
    graph_root_format = re.compile(r'\([^\n]+')
    graph_format=re.compile(r'(\t+)[^\n]+')
    short_alignment=re.compile(r'# ::short\t([^\n]+)\t')
    gs = []
    documental_amrs=[]
    for amrid,amr in enumerate(amrs):
        lines=amr.split('\n')
        tokens = lines[0]
        assert tokens.startswith('# ::tok ')
        tokens = tokens[len("# ::tok "):].split(' ')

        for lid,line in enumerate(lines):
            if line.startswith('# ::short'):
                amr_start_line=lid+1
                break
        amrgraph=''.join(lines[amr_start_line:])
        amrgraph=re.sub(r'\s{4,}','',amrgraph)
        # amrgraph=re.sub(r'/\s[\w\-]+','',amrgraph)
        # amrgraph=amrgraph.split(':')

        nodes = node_format.findall(amr)
        edges = edge_format.findall(amr)

        short_alignments = short_alignment.findall(amr)[0]
        assert short_alignments[0]=='{' and short_alignments[-1]=='}'
        short_alignments=re.sub(r'(\d+):',r'"\1":',short_alignments)
        short_alignments=re.sub(r'\'',r'"',short_alignments)
        short_alignments=json.loads(short_alignments)

        documental_amrs.append((nodes, edges, tokens, amrgraph, short_alignments))
        if (amrid+1)%5==0: ##TODO sentence number of a document may not be 5
            graph = DocumentalAmrGraph(documental_amrs)
            documental_amrs=[]
            gs.append(graph)
    return gs

def load_amrs_to_graph(amrs:List[str])->DocumentalAmrGraph:
    node_format = re.compile(r'# ::node\t(\S+)\t(\S+)\t(\d+)-(\d+)')
    edge_format = re.compile(r'# ::edge\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)')
    graph_root_format = re.compile(r'\([^\n]+')
    graph_format=re.compile(r'\s+:\S+\s \((\S+)')
    short_alignment=re.compile(r'# ::short\t([^\n]+)\t')
    documental_amrs = []
    for amrid, amr in enumerate(amrs):
        lines=amr.split('\n')
        tokens = lines[0]
        assert tokens.startswith('# ::tok ')
        tokens = tokens[len("# ::tok "):].split(' ')

        for lid,line in enumerate(lines):
            if line.startswith('# ::short'):
                amr_start_line=lid+1
                break
        amrgraph = ''.join(lines[amr_start_line:])
        amrgraph = re.sub(r'\s{4,}', '', amrgraph)

        nodes = node_format.findall(amr)
        edges = edge_format.findall(amr)

        short_alignments = short_alignment.findall(amr)[0]
        assert (short_alignments[0] == '{' and short_alignments[-1] == '}')
        short_alignments = re.sub(r'(\d+):', r'"\1":', short_alignments)
        short_alignments = re.sub(r'\'', r'"', short_alignments)
        short_alignments = json.loads(short_alignments)

        documental_amrs.append((nodes, edges, tokens,amrgraph,short_alignments))
    graph = DocumentalAmrGraph(documental_amrs)

    return graph

def parse_RAMS(filename):
    parser = AMRParser.from_checkpoint(
        '/mnt/ssd/amr/transition-amr-parser-action-pointer/DATA/AMR2.0/models/exp_cofill_o8.3_act-states_RoBERTa-large-top24/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_wiki.smatch_top3-avg.pt')
    gs=[]
    with open(filename) as f:
        all_dicts=f.readlines()
        for line in all_dicts:
            instance = json.loads(line)
            annotations = parser.parse_sentences(instance['sentences'])
            graph=load_amrs_to_graph(annotations[0])
            gs.append(graph)
    return gs

if __name__=="__main__":
    gs=load_document_amr('./test.amr.txt')
    G=gs[0]
    G.merge_coref(coref_info)
    start_node=(51,52)
    end_node=(26,31)
    path = G.BFS(start_node,end_node)
    print(path)
    tokens = sum(G.tokens,[])
    path_text=[]
    for node in path:
        if node==[]:continue
        node_text=[]
        for n in node:
            if n==-1:continue
            node_text.append(' '.join(tokens[n[0][0]:n[0][1]]))
        path_text.append(node_text)
    print(path_text)
    print(G.subgraph(path))
