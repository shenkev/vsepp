from vocab import Vocabulary
import evaluation
from evaluation import ImageRetriever
from annoy import AnnoyIndex
import time
import torch

r = ImageRetriever('/media/shenkev/data/Ubuntu/Project/vsepp/runs/coco_vse++/model_best.pth.tar',
                   '/media/shenkev/data/Ubuntu/Project/Data',
                   '/media/shenkev/data/Ubuntu/Project/vsepp/data/data',
                   'train',
                   '/media/shenkev/data/Ubuntu/Project/vsepp/vocab/coco_precomp_vocab.pkl', 128)

topk = 20
# query = 'A little girl is getting ready to blow out a candle on a small dessert.'
query = 'a long table with a plant on top of it surrounded with wooden chairs'
inds, q_emb, img_emb = r.get_NN(query, measure='dot', k=topk)

r.visualise_NN(inds, 'exa', "exactNearestNeighbours.html")

trees = 10
z_dim = img_emb.shape[1]
t = AnnoyIndex(z_dim)
for i in xrange(img_emb.shape[0]):
    t.add_item(i, img_emb[i])

t.build(trees)

tic = time.clock()
ann_inds = t.get_nns_by_vector(q_emb.squeeze(), topk, search_k=-1)
toc = time.clock()
ann_inds = [i * 5 for i in ann_inds]
print ('ANN search took {} ms over {} images'.format((toc - tic) * 1000.0, img_emb.shape[0]))

r.visualise_NN(ann_inds, 'app', "approximateNearestNeighbours.html")

# import pickle
# from data import get_test_loader
#
# vocab_path = '/media/shenkev/data/Ubuntu/Project/vsepp/vocab/coco_vocab.pkl'
# # load vocabulary used by the model
# with open(vocab_path, 'rb') as f:
#     vocab = pickle.load(f)
# vocab_size = len(vocab)
#
# # load model and options
# checkpoint = torch.load('/media/shenkev/data/Ubuntu/Project/vsepp/runs/coco_vse++/model_best.pth.tar')
# opt = checkpoint['opt']
# opt.data_path = '/media/shenkev/data/Ubuntu/Project/vsepp/data/data'
# data_loader = get_test_loader('test', opt.data_name, vocab, 224,
#                               32, 1, opt)
#
# dir = "coco_vse++"
# img_embs, cap_embs, dataset = evaluation.evalrank("/media/shenkev/data/Ubuntu/Project/vsepp/runs/{}/model_best.pth.tar".format(dir),
#                     vocab_path='./vocab/', data_path="/media/shenkev/data/Ubuntu/Project/vsepp/data/data", split="test")


