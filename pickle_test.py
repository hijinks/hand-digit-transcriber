import numpy as np

danger_file = r'/run/media/sam/3086-05D9/Teach_Data/G10/A/11/fines_learn_data.npy'
try:
    d = np.load(danger_file)
except (OSError, IOError, EOFError) as e:
    print 'nope'
