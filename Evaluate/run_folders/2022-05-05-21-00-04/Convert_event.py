import h5py
import numpy as np

f = h5py.File('/home/zzy/rpg_vid2e/esim_py/tests/raw_events.h5','r')
events = f['events']

w = events[:,0].astype(np.int_)
h = events[:,1].astype(np.int_)
t = events[:,2]*1000
t = t.astype(np.int_)
act = events[:,3].astype(np.int_)
f.close()
event = np.concatenate([w[:,np.newaxis],h[:,np.newaxis],t[:,np.newaxis],act[:,np.newaxis]],axis=1)
f = h5py.File('/home/zzy/rpg_vid2e/esim_py/tests/recon_events.h5','w')
f.create_dataset('events',data = event)
f.close()