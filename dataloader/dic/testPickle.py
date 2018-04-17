import pickle

with open('./frame_count.pickle','rb') as file:
     dic_frame = pickle.load(file)
     print dic_frame
     print len(dic_frame.keys())
file.close()
