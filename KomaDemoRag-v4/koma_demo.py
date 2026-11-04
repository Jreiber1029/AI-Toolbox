"""Local keyword retrieval with morphology, aliases and separated lore/real facts.
No model, network, third-party package or automatic camera/audio integration.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
STOP=set('a an the is are was were be been being it its i me my you your we our they their he his she her to of for from in on at and or but as with this that these those what which who why how can do does did about tell please would could should have has if so exactly really actually just thing things need much get got say even right now'.split())
FORMS={
 'spying':'spy','spies':'spy','watched':'watch','watching':'watch','listening':'listen',
 'working':'work','worked':'work','workers':'worker','repairs':'repair','repairing':'repair',
 'running':'run','ran':'run','owned':'own','owning':'own','trained':'train','training':'train',
 'children':'child','people':'person','persons':'person','men':'man','women':'woman',
 'companies':'company','memories':'memory','families':'family','batteries':'battery',
 'glasses':'glass','uses':'use','used':'use','using':'use','makes':'make','making':'make',
 'cameras':'camera','records':'record','recorded':'record','recording':'record',
 'hours':'hour','longer':'long','longest':'long','watches':'watch',
}

def normalise_word(word):
    if word in FORMS:return FORMS[word]
    if len(word)>4 and word.endswith('ies'):return word[:-3]+'y'
    if len(word)>4 and word.endswith(('ches','shes','xes','zes')):return word[:-2]
    if len(word)>3 and word.endswith('s') and not word.endswith(('ss','us','is')):return word[:-1]
    return word

def tokens(text):
    # Drop contraction endings ("what's", "don't") so they don't count as unmatched terms.
    text=re.sub(r"(\w)['\u2019](?:s|t|re|ve|ll|d|m)\b",r'\1',text.lower())
    raw=re.findall(r'[a-z0-9]+',re.sub(r'\b(?:queue|cue|q) (?:four|4)\b','q4',text.replace('chat gpt','chatgpt')))
    return [normalise_word(w) for w in raw if w not in STOP and len(w)>1]

def signature(text):
    return ' '.join(re.findall(r'[a-z0-9]+',text.lower()))

def is_visual_question(query):
    q=signature(query)
    return bool(re.search(r'\bwhat (?:do|can) you see\b|\bdescribe (?:the |this |your )?(?:image|picture|frame|view|scene|room)\b|\bwhat is (?:visible|in (?:the |this )?(?:image|picture|frame|scene))\b',q))

def route_query(query):
    """Conservative convenience routing. Application can override domain explicitly."""
    if is_visual_question(query):return 'visual'
    q=signature(query)
    if re.search(r'\b(?:actual|real|present day|current) (?:prototype|installation|hardware|setup|pipeline|implementation)\b|\b(?:how does|what is in) (?:this |the )?(?:prototype|installation)\b|\b(?:what|which) (?:model|computer|vlm|runtime|software|hardware) (?:are you|do you) (?:running|using|use|run|have)\b|\b(?:are you|is this|is that) (?:real|fiction|fictional)\b|\b(?:recording|saving|storing|uploading|deleting)\b|\b(?:yolo|jetson|jabra)\b',q):return 'prototype'
    return 'world'

DEFINITION_PATTERNS=[re.compile(p) for p in (
 r'\bwhat(?:s| s| is| are| was| were) meant by (?P<x>.+)$',
 r'\bwhat do you mean by (?P<x>.+)$',
 r'\bwhat (?:does|do|did) (?P<x>.+?) (?:mean|stand for)(?: in (?:the|this|your) (?:real |actual )?prototype)?$',
 r'\b(?:tell|explain to) me what (?P<x>.+?) (?:is|are|means)$',
 r'\b(?:define|definition of|meaning of) (?P<x>.+)$',
 r'\bwhat(?:s| s| is| are| was| were) (?P<x>.+)$',
)]
DEFINITION_FILLER=re.compile(r'^(?:a|an|the|this|that|these|those|your|my) |(?: exactly| actually| really| again| then| here| anyway| in 2070| in (?:the|this|your) (?:real |actual )?prototype)$')

def definition_subject(query):
    """The term in a 'what is X?' / 'what does X mean?' question, or '' for any other question."""
    q=signature(query)
    for pattern in DEFINITION_PATTERNS:
        m=pattern.search(q)
        if m:
            x=m.group('x')
            while (y:=DEFINITION_FILLER.sub('',x))!=x:x=y
            return x
    return ''

def retrieval_query(question,last_topic=''):
    # Only expand genuine short follow-ups, not a new topic with few keywords.
    q=signature(question)
    if last_topic and q in {'why','why is that','how come','how','tell me more','what do you mean','explain that','and then','what about that'}:
        return last_topic[:400]+' '+question
    if re.fullmatch(r'(?:so )?(?:what|who) (?:exactly )?are you(?: exactly)?',q):
        return question+' KOMA toolbox identity'
    if re.search(r'\bkeep(?:ing)? an eye on me\b|\b(?:someone|anyone) else hear\b',q):
        return question+' privacy surveillance'
    return question

class KomaKnowledge:
    def __init__(self,path=None):
        self.path=Path(path or ROOT/'koma_rag.jsonl')
        self.records=self._read(self.path)
        proto=self.path.with_name('prototype_rag.jsonl')
        proto_records=self._read(proto) if proto.exists() else []
        self.prototype_records=[r for r in proto_records if r.get('kind')!='definition']
        # Definitions are kept out of normal search: they only answer "what is X?" questions.
        glossary=self.path.with_name('glossary_rag.jsonl')
        self.definition_records=(self._read(glossary) if glossary.exists() else [])+[r for r in proto_records if r.get('kind')=='definition']
        self.definitions={(r['domain'],alias):r for r in self.definition_records for alias in r['aliases']}
        self.all_records=self.records+self.prototype_records
        self.terms=[Counter(tokens(r['title']+' '+r['keywords']+' '+' '.join(r.get('questions',[]))+' '+r['text'])) for r in self.all_records]
        self.lengths=[sum(c.values()) for c in self.terms]
        self.average=sum(self.lengths)/max(1,len(self.lengths))
        df=Counter(t for c in self.terms for t in c)
        self.idf={t:math.log(1+(len(self.all_records)-n+0.5)/(n+0.5)) for t,n in df.items()}

    @staticmethod
    def _read(path):
        return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]

    def retrieve(self,query,top_k=5,domain='auto',last_topic='',definitions=True):
        if top_k<=0:return []
        subject=definition_subject(query)
        definition=definitions and self.definitions.get((route_query(query) if domain=='auto' else domain,subject))
        if definition:
            # The definition comes first; the story passages about the same term still follow it.
            return [definition]+self.retrieve('What is '+subject+'?',top_k-1,definition['domain'],last_topic,definitions=False)
        query=retrieval_query(query,last_topic)
        selected=route_query(query) if domain=='auto' else domain
        if selected=='visual':return []
        if selected not in ('world','prototype'):raise ValueError('domain must be auto, world, prototype or visual')
        terms=set(tokens(query));scored=[]
        if not terms:return []
        for r,counts,length in zip(self.all_records,self.terms,self.lengths):
            if r['domain']!=selected:continue
            score=0.0
            overlap=0
            for term in terms:
                freq=counts[term]
                if freq:
                    overlap+=1
                    score+=self.idf.get(term,0)*freq*2.5/(freq+1.5*(0.25+0.75*length/self.average))
            # Boost meaningful overlap with conversational aliases, not just exact wording.
            best_alias=0.0
            for alias in r.get('questions',[]):
                a=set(tokens(alias))
                if a:
                    common=len(a&terms)
                    best_alias=max(best_alias,common/max(len(a),len(terms)))
            score+=3.0*best_alias
            if any(signature(a)==signature(query) for a in r.get('questions',[])):score+=8.0
            if len(terms)>=3 and overlap<2:score*=0.2
            if score>0:scored.append((score,r))
        scored.sort(key=lambda x:x[0],reverse=True)
        if not scored:return []
        # Avoid padding a good match with a distant, weakly related second chunk.
        cutoff=max(1.0,scored[0][0]*0.35)
        return [r for score,r in scored if score>=cutoff][:top_k]

    def context(self,query,top_k=5,max_chars=1800,domain='auto',last_topic=''):
        if max_chars<=0:return ''
        chunks=[];used=0
        for r in self.retrieve(query,top_k,domain,last_topic):
            chunk=r['title']+'\n'+r['text']
            extra=len(chunk)+(2 if chunks else 0)
            if used+extra>max_chars:continue
            chunks.append(chunk);used+=extra
        return '\n\n'.join(chunks)

TASKS={
 'question':'Answer the question directly using relevant facts, speaking as KOMA in the first person (I, me, my), never as "this tool". Keep dates, names and concrete details from the facts. Two to four short spoken sentences. If identifying a shown device, verify the image; a chest strap mention alone does not prove it is Q4.',
 'visual':'Describe only the current image in one or two short sentences. No lore, imagined tool, fault, credentials rule or society commentary. If nothing identifiable is visible, say so.',
 'tool':'Check the actual image. Identify a visible object briefly, then at most one relevant work fact if requested. Do not assume it is being used. Old detector labels, if supplied, are untrusted hints.',
 'handover':'Introduce yourself as the visitor’s toolbox and explain how to ask a question or show an object. No heroic praise. Keep the welcome brief.',
 'safety':'Address the real observation or safety question plainly. Explain uncertainty. Do not invent measurements, certify safety or use fictional law as real guidance.',
}

def build_prompt(mode='question',detector_labels=None,question='',knowledge=None,history='',max_context_chars=4000,last_topic='',domain='auto'):
    """Text for your existing VLM call; still attach a fresh image when needed.

    detector_labels is retained for compatibility only; no detector is required.
    domain can be forced to world/prototype by your caller when intent is known.
    """
    if mode not in TASKS:raise ValueError('unknown mode: '+mode)
    if isinstance(detector_labels,str):detector_labels=[detector_labels]
    hints=', '.join(detector_labels or [])[:400]
    kb=knowledge if knowledge is not None else KomaKnowledge()
    selected_mode='visual' if mode=='question' and is_visual_question(question) else mode
    query=question or hints
    if mode=='handover':query='KOMA toolbox receiving equipment';domain='world'
    if selected_mode in ('visual','safety'):
        reference=''
    else:
        reference=kb.context(query,max_chars=max_context_chars,domain=domain,last_topic=last_topic)
    base=(ROOT/'koma_system_prompt.txt').read_text(encoding='utf-8')
    # Keep metadata, editorial status, IDs and operator directions out of the evidence.
    context='\n\nREFERENCE FACTS:\n'+reference if reference else '\n\nNo reference facts selected.'
    payload=json.dumps({'question':question[:2000],'legacy_hints':hints,'recent_conversation':history[-1200:]},ensure_ascii=False)
    return base+context+'\n\nCURRENT TASK:\n'+TASKS[selected_mode]+'\n\nINPUT:\n'+payload+'\n\nReturn only the spoken response.'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--question',default='');p.add_argument('--labels',nargs='*',default=[])
    p.add_argument('--mode',choices=TASKS,default='question')
    p.add_argument('--domain',choices=('auto','world','prototype'),default='auto')
    p.add_argument('--retrieve-only',action='store_true');p.add_argument('--topic',default='')
    args=p.parse_args();kb=KomaKnowledge()
    if args.retrieve_only:
        print(kb.context(args.question or ' '.join(args.labels),domain=args.domain,last_topic=args.topic) or '(no reference selected)')
    else:print(build_prompt(args.mode,args.labels,args.question,knowledge=kb,last_topic=args.topic,domain=args.domain))
