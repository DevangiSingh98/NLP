#!/usr/bin/env python3
"""Hindi POS tagging: Naive Bayes and three HMM decoding strategies.
Input format: one `word<TAB>tag` (or whitespace-separated word/tag) per line;
blank lines delimit sentences. Run: python hindi_pos_tagger.py --data-dir . --out-dir results
"""
import argparse, json, math, os, csv, time
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, classification_report

FEATURES = ('length', 'word', 'prev_word', 'next_word', 'prev_tag')

def read_data(path):
    sentences, current = [], []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if current: sentences.append(current); current=[]
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2: continue
            current.append((parts[0], parts[1]))
    if current: sentences.append(current)
    return sentences

def features(words, i, prev_tag):
    w = words[i]
    return {'length': 'MORE' if len(w) > 4 else 'LESS',
            'word': w,
            'prev_word': words[i-1] if i else '<BOS>',
            'next_word': words[i+1] if i+1 < len(words) else '<EOS>',
            'prev_tag': prev_tag if prev_tag is not None else '<BOS_TAG>'}

class NaiveBayesTagger:
    """Multinomial categorical NB; prev_tag is the preceding gold tag in training
    and the preceding predicted tag at inference (left-to-right greedy decoding)."""
    def __init__(self, alpha=0.1):
        self.alpha=alpha
    def fit(self, sentences):
        self.tags=sorted({t for s in sentences for _,t in s})
        self.tag_counts=Counter(); self.feature_counts={f:defaultdict(Counter) for f in FEATURES}
        self.feature_totals={f:Counter() for f in FEATURES}; self.vocab={f:set() for f in FEATURES}
        for sent in sentences:
            words=[w for w,t in sent]; prev=None
            for i,(w,t) in enumerate(sent):
                self.tag_counts[t]+=1
                for fn,val in features(words,i,prev).items():
                    self.feature_counts[fn][t][val]+=1
                    self.feature_totals[fn][t]+=1; self.vocab[fn].add(val)
                prev=t
        self.n=sum(self.tag_counts.values()); self.log_prior={t:math.log(self.tag_counts[t]/self.n) for t in self.tags}
        self.log_den={f:{t:math.log(self.feature_totals[f][t]+self.alpha*(len(self.vocab[f])+1)) for t in self.tags} for f in FEATURES}
        return self
    def predict_sentence(self, words):
        out=[]; prev=None
        for i,w in enumerate(words):
            vals=features(words,i,prev); best=None; best_score=-float('inf')
            for t in self.tags:
                score=self.log_prior[t]
                for fn,val in vals.items():
                    score += math.log(self.feature_counts[fn][t][val]+self.alpha)-self.log_den[fn][t]
                if score>best_score: best_score=score; best=t
            out.append(best); prev=best
        return out
    def predict(self, sentences): return [self.predict_sentence([w for w,t in s]) for s in sentences]

class HMMTagger:
    def __init__(self, alpha=0.1): self.alpha=alpha
    def fit(self, sentences):
        self.tags=sorted({t for s in sentences for _,t in s}); self.K=len(self.tags); self.idx={t:i for i,t in enumerate(self.tags)}
        self.vocab=sorted({w for s in sentences for w,t in s}); self.widx={w:i for i,w in enumerate(self.vocab)}; self.V=len(self.vocab)
        starts=np.zeros(self.K); trans=np.zeros((self.K,self.K)); emits=np.zeros((self.K,self.V)); ends=np.zeros(self.K)
        for sent in sentences:
            ids=[self.idx[t] for w,t in sent]
            if not ids: continue
            starts[ids[0]]+=1; ends[ids[-1]]+=1
            for j,(w,t) in enumerate(sent):
                emits[ids[j],self.widx[w]]+=1
                if j: trans[ids[j-1],ids[j]]+=1
        a=self.alpha
        self.log_start=np.log((starts+a)/(starts.sum()+a*self.K))
        self.log_end=np.log((ends+a)/(ends.sum()+a*self.K))
        self.log_trans=np.log((trans+a)/(trans.sum(axis=1,keepdims=True)+a*self.K))
        self.log_emit=np.log((emits+a)/(emits.sum(axis=1,keepdims=True)+a*(self.V+1)))
        self.log_unk=np.log(a/(emits.sum(axis=1)+a*(self.V+1)))
        return self
    def emission(self, words):
        out=np.empty((len(words),self.K))
        for i,w in enumerate(words):
            wi=self.widx.get(w)
            out[i]=self.log_emit[:,wi] if wi is not None else self.log_unk
        return out
    def viterbi(self, words):
        if not words: return []
        E=self.emission(words); n=len(words); dp=np.empty((n,self.K)); back=np.zeros((n,self.K),dtype=int)
        dp[0]=self.log_start+E[0]
        for i in range(1,n):
            scores=dp[i-1][:,None]+self.log_trans
            back[i]=np.argmax(scores,axis=0); dp[i]=np.max(scores,axis=0)+E[i]
        state=int(np.argmax(dp[-1]+self.log_end)); path=[state]
        for i in range(n-1,0,-1): path.append(int(back[i,path[-1]]))
        return [self.tags[j] for j in path[::-1]]
    def backward_viterbi(self, words):
        """Max-sum suffix DP, then recover the best path from the start."""
        if not words: return []
        E=self.emission(words); n=len(words)
        beta=np.empty((n,self.K)); choice=np.zeros((n-1,self.K),dtype=int)
        beta[-1]=self.log_end
        for i in range(n-2,-1,-1):
            scores=self.log_trans+E[i+1][None,:]+beta[i+1][None,:]
            choice[i]=np.argmax(scores,axis=1); beta[i]=np.max(scores,axis=1)
        state=int(np.argmax(self.log_start+E[0]+beta[0])); path=[state]
        for i in range(n-1): path.append(int(choice[i,path[-1]]))
        return [self.tags[j] for j in path]
    def forward_backward(self, words):
        """Forward-backward posterior decoding: choose highest marginal-probability
        tag at each token. This maximizes token accuracy, not whole-sequence score."""
        if not words: return []
        E=self.emission(words); n=len(words)
        alpha=np.empty((n,self.K)); beta=np.empty((n,self.K))
        alpha[0]=self.log_start+E[0]
        for i in range(1,n):
            z=alpha[i-1][:,None]+self.log_trans
            m=np.max(z,axis=0); alpha[i]=E[i]+m+np.log(np.exp(z-m[None,:]).sum(axis=0))
        beta[-1]=self.log_end
        for i in range(n-2,-1,-1):
            z=self.log_trans+E[i+1][None,:]+beta[i+1][None,:]
            m=np.max(z,axis=1); beta[i]=m+np.log(np.exp(z-m[:,None]).sum(axis=1))
        logz=np.logaddexp.reduce(alpha[-1]+self.log_end)
        posterior=alpha+beta-logz
        return [self.tags[int(np.argmax(row))] for row in posterior]
    def predict(self, sentences, method):
        fn={'viterbi':self.viterbi,'backward_viterbi':self.backward_viterbi,'forward_backward':self.forward_backward}[method]
        return [fn([w for w,t in s]) for s in sentences]

def evaluate(name, gold_sents, pred_sents, tags):
    ytrue=[t for s in gold_sents for w,t in s]; ypred=[t for s in pred_sents for t in s]
    labels=sorted(set(tags)|set(ytrue)|set(ypred))
    scores=f1_score(ytrue,ypred,labels=labels,average=None,zero_division=0)
    per={t:float(v) for t,v in zip(labels,scores)}
    return {'algorithm':name,'accuracy':float(accuracy_score(ytrue,ypred)),
            'macro_f1':float(f1_score(ytrue,ypred,labels=labels,average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(ytrue,ypred,labels=labels,average='weighted',zero_division=0)),
            'per_tag_f1':per,'labels':labels,'ytrue':ytrue,'ypred':ypred}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='.'); ap.add_argument('--out-dir',default='hindi_pos_results'); ap.add_argument('--alpha',type=float,default=0.1)
    args=ap.parse_args(); os.makedirs(args.out_dir,exist_ok=True)
    # The provided assignment files are sometimes downloaded with a suffix,
    # e.g. ``hindi-train-pos (1).txt``. Prefer the canonical name, then accept
    # exactly one matching suffixed file in the selected data directory.
    paths={}
    for split in ('train','dev','test'):
        canonical=os.path.join(args.data_dir,f'hindi-{split}-pos.txt')
        matches=sorted(p for p in __import__('glob').glob(os.path.join(args.data_dir,f'hindi-{split}-pos*.txt')) if os.path.isfile(p))
        if os.path.isfile(canonical):
            paths[split]=canonical
        elif len(matches)==1:
            paths[split]=matches[0]
        elif not matches:
            ap.error(f"Missing {split} data: expected {canonical} (or hindi-{split}-pos*.txt) in --data-dir")
        else:
            ap.error(f"Multiple {split} files match hindi-{split}-pos*.txt; rename the intended split to hindi-{split}-pos.txt")
    if args.alpha <= 0:
        ap.error('--alpha must be greater than zero')
    train,dev,test=[read_data(paths[x]) for x in ('train','dev','test')]
    print('Sentences / tokens:', {k:(len(v),sum(len(s) for s in v)) for k,v in [('train',train),('dev',dev),('test',test)]},flush=True)
    models=[]
    t0=time.time(); nb=NaiveBayesTagger(args.alpha).fit(train); print('Trained Naive Bayes in',round(time.time()-t0,1),'s',flush=True)
    t0=time.time(); hmm=HMMTagger(args.alpha).fit(train); print('Trained HMM in',round(time.time()-t0,1),'s',flush=True)
    for split_name, data in [('dev',dev),('test',test)]:
        print('Predicting',split_name,flush=True)
        predictions={'Naive Bayes':nb.predict(data)}
        for method,label in [('viterbi','HMM forward Viterbi'),('backward_viterbi','HMM backward Viterbi'),('forward_backward','HMM forward-backward posterior decoding')]:
            predictions[label]=hmm.predict(data,method)
        gold_tags=sorted(hmm.tags)
        for name,pred in predictions.items():
            r=evaluate(name,data,pred,gold_tags); models.append((split_name,r))
            print(f"{split_name:4} | {name:42} accuracy={r['accuracy']:.4f} macro-F1={r['macro_f1']:.4f} weighted-F1={r['weighted_f1']:.4f}",flush=True)
            with open(os.path.join(args.out_dir,f'{split_name}_{name.lower().replace(" ","_").replace("-","_")}_report.txt'),'w',encoding='utf-8') as f:
                f.write(classification_report(r['ytrue'],r['ypred'],labels=r['labels'],zero_division=0,digits=4))
        # predictions per tag table
        with open(os.path.join(args.out_dir,f'{split_name}_per_tag_f1.csv'),'w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['tag']+[r['algorithm'] for sn,r in models if sn==split_name])
            split_results=[r for sn,r in models if sn==split_name]
            for tag in sorted(set(t for s in data for w,t in s)):
                writer.writerow([tag]+[f"{r['per_tag_f1'].get(tag,0):.6f}" for r in split_results])
    # Summary CSV
    with open(os.path.join(args.out_dir,'summary_metrics.csv'),'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['split','algorithm','accuracy','macro_f1','weighted_f1'])
        for split,r in models: writer.writerow([split,r['algorithm'],r['accuracy'],r['macro_f1'],r['weighted_f1']])
    # Charts: test per-tag and overall metrics for dev/test
    test_results=[r for sn,r in models if sn=='test']; tags=sorted(set(t for s in test for w,t in s))
    x=np.arange(len(tags)); width=0.8/len(test_results)
    fig,ax=plt.subplots(figsize=(16,6))
    for i,r in enumerate(test_results): ax.bar(x+(i-(len(test_results)-1)/2)*width,[r['per_tag_f1'].get(t,0) for t in tags],width,label=r['algorithm'])
    ax.set_xticks(x); ax.set_xticklabels(tags,rotation=55,ha='right'); ax.set_ylim(0,1.02); ax.set_ylabel('F1-score'); ax.set_title('Hindi POS tagging: per-tag F1 on test split'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(args.out_dir,'test_per_tag_f1.png'),dpi=180); plt.close(fig)
    algs=[r['algorithm'] for r in [r for sn,r in models if sn=='dev']]
    metrics=['accuracy','macro_f1','weighted_f1']; fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,metric in zip(axes,metrics):
        vals=[]
        for split in ('dev','test'):
            vals.append([r[metric] for sn,r in models if sn==split])
        pos=np.arange(len(algs)); w=.36
        ax.bar(pos-w/2,vals[0],w,label='Dev'); ax.bar(pos+w/2,vals[1],w,label='Test')
        ax.set_xticks(pos); ax.set_xticklabels(['NB','Forward\nViterbi','Backward\nViterbi','Forward-Backward'],rotation=0); ax.set_ylim(0,1.02); ax.set_title(metric.replace('_',' ').title()); ax.grid(axis='y',alpha=.25)
        if ax is axes[0]: ax.legend()
    fig.suptitle('Overall performance comparison'); fig.tight_layout(); fig.savefig(os.path.join(args.out_dir,'overall_metrics.png'),dpi=180); plt.close(fig)
    # Machine-readable complete per-tag metrics
    with open(os.path.join(args.out_dir,'all_per_tag_f1.csv'),'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['split','algorithm','tag','f1'])
        for split,r in models:
            for tag in tags: writer.writerow([split,r['algorithm'],tag,r['per_tag_f1'].get(tag,0)])
    with open(os.path.join(args.out_dir,'run_info.json'),'w',encoding='utf-8') as f:
        json.dump({'alpha':args.alpha,'train_sentences':len(train),'dev_sentences':len(dev),'test_sentences':len(test),'train_tokens':sum(len(s) for s in train),'dev_tokens':sum(len(s) for s in dev),'test_tokens':sum(len(s) for s in test),'tags':hmm.tags},f,ensure_ascii=False,indent=2)
    print('Wrote results to',args.out_dir,flush=True)
if __name__=='__main__': main()
