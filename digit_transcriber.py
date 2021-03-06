import cv2
import scipy
import PIL
from PIL import ImageTk
import os
from Tkinter import *
from scipy import ndimage as nd
from math import degrees, atan2
import re
import csv
import ruamel.yaml
import numpy as np
import threading
from shutil import copyfile
import random
import itertools
from itertools import cycle
from cement.core import foundation, controller
from cement.core.controller import expose
from cement.utils import shell

from sklearn.externals import joblib
from skimage.feature import hog

class DigitAppController(controller.CementBaseController):
    class Meta:
        label = 'base'
        description = 'Python CLI application to transcribe Wolman point counts'

    @expose(hide=True, aliases=['run'])
    def default(self):
        print("Setting up...")

class DigitApp(foundation.CementApp):

    class Meta:
        label = 'Digit_Controller'
        base_controller = DigitAppController

app = DigitApp()

class Transcriber:
    'Common base class for capturing and recognising hand-written digits'

    def __init__(self, config):
        self.meta = None
        self.images_path = None
        self.output_path = None
        self.data_path = None
        self.taught_data = []
        self.taught_labels = []
        self.x_points = []
        self.y_points = []
        self.point_ids = []
        self.number_input = {}
        self.check_workspace(config)

        rects, im_blank, im, im_th, image_choice, im_check = self.load_image()

        self.get_meta(image_choice)

        im_points, im_check, im_blank, input_numbers = self.transcribe(rects, im, im_th, im_blank, im_check)

        points = np.array([self.point_ids, self.x_points, self.y_points])

        numbers = self.connect_digits(points, im_points)

        self.save_data(numbers, image_choice, im_check, im_blank, input_numbers)

    def check_workspace(self, config):
        # Set where we typically source our images, set output directory

        if os.path.isdir(config['image_directory']):
            self.images_path = config['image_directory']

        else:
            while self.images_path is None:
                p = shell.Prompt("Path to images: ")
                if os.path.isdir(p.input):
                    self.images_path = p.input
        if os.path.isdir(config['output_directory']):
            self.output_path = config['output_directory']
        else:
            while self.output_path is None:
                p = shell.Prompt("Path to output: ")
                if os.path.isdir(p.input):
                    self.output_path = p.input

        if os.path.isdir(config['data_directory']):
            self.data_path = config['data_directory']

        # Save config
        config = {
            'digit_recogniser': {
                'image_directory': self.images_path,
                'output_directory': self.output_path,
                'data_directory': self.data_path
            }
        }
        ruamel.yaml.dump(config, open(sam_config_path, 'w'), Dumper=ruamel.yaml.RoundTripDumper)

    def get_meta(self, image_choice):
        # Context and data labels
        fan = None
        surface = None
        site = None
        name = None
        cover = None
        location = None
        meta_check = False
        if self.data_path:
            image_name = os.path.splitext(os.path.basename(image_choice))[0]
            print image_name
            print 'wolman_'+image_name+'.yml'
            if os.path.exists(os.path.join(self.data_path, 'wolman_'+image_name+'.yml')):

                image_data = ruamel.yaml.load(open(os.path.join(self.data_path, 'wolman_'+image_name+'.yml')), ruamel.yaml.RoundTripLoader)
                fan = image_data['fan']
                surface = image_data['surface']
                site = image_data['site']
                name = image_data['name']
                cover = image_data['cover']
                location = image_data['location']
                meta_check = True

        if meta_check is False:

            while fan is None:
                p = shell.Prompt("Fan: ")
                if p:
                    fan = p.input

            while surface is None:
                p = shell.Prompt("Surface: ")
                if p:
                    surface = p.input

            while site is None:
                p = shell.Prompt("Site: ")
                if p:
                    site = p.input

            while name is None:
                p = shell.Prompt("Name: ")
                if p:
                    name = p.input

            while cover is None:
                p = shell.Prompt("% cover: ")
                if p:
                    cover = p.input

            while location is None:
                p = shell.Prompt("Location: ")
                if p:
                    location = p.input

        self.meta = {
            'fan': fan,
            'surface': surface,
            'site': site,
            'cover': cover,
            'name': name,
            'location': location
        }

    def load_image(self):
        # Load image and prepare it for analysis
        image_choice = None
        while image_choice is None:
            p = shell.Prompt("Image: ")
            print 'Checking '+self.images_path
            if os.path.exists(os.path.join(self.images_path, p.input)):
                image_choice = os.path.join(self.images_path, p.input)
            else:
                print 'Could not find '+p.input

        im = cv2.imread(image_choice)
        height, width, channels = im.shape


        # Convert to grayscale and apply Gaussian filtering
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
        im_blank = np.asarray(PIL.Image.new("RGB", (width, height), "white"))
        im_check = im_blank.copy()
        im_blank = im.copy()
        ret, im_blank = cv2.threshold(im_blank, 90, 255, cv2.THRESH_BINARY_INV)

        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        return rects, im_blank, im, im_th, image_choice, im_check

    def select_label(self, l_array, label):
        mne = np.ma.masked_not_equal(l_array, label)
        inverted = np.invert(mne.mask)
        return (inverted * 255).astype(np.uint8)

    def transcribe(self, rects, im, im_th, im_blank, im_check):
        # Use human to tell us what digits are

        i = 0
        im_orig = im.copy()
        im_points = im.copy()
        input_numbers = []
        s_type = len(rects)

        data = {}
        mistakes = []

        for rect in rects:
            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            cv2.rectangle(im_blank, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

            s_type = s_type - 1
            print s_type

            if pt1 < 0:
                pt1 = 0

            if pt2 < 0:
                pt2 = 0

            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            s = roi.shape

            roi_x_org = int(s[0]/2)
            roi_y_org = int(s[1]/2)

            i = i +1

            #if i > 10:
                #break;


            if roi.size > 100:
                labeled_array, num_features = nd.label(roi)
                loc = nd.find_objects(labeled_array)
                sums = []
                labels = range(1, num_features+1)

                for l in loc:
                    a = labeled_array[l]
                    am = np.ma.masked_greater(a, 0)
                    sums.append(scipy.sum(am.mask))

                # Sum of locations, whichever has the largest is our target
                target_label = sums. index(max(sums))+1
                roi = self.select_label(labeled_array, target_label)

                x_org = rect[0] + (rect[2]/2)
                y_org = rect[1] + (rect[3]/2)

                digit_loc = im_orig.copy()
                cv2.rectangle(digit_loc, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
                dh, dw, dc = digit_loc.shape
                hog_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi_hog_fd = hog(hog_roi, orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
                guess = clf.predict(np.array([roi_hog_fd], dtype=float))[0]


                digit_loc = cv2.resize(digit_loc, (int(dw*sf2), int(dh*sf2)))

                input_number = False
                #input_number = random.randint(0,9)

                label_choice = cycle(labels)
                v = label_choice.next()

                while input_number is False:
                    rh, rw = roi.shape
                    iP = imagePrompt(roi, rh, rw, guess)
                    if iP.output == 'check':
                        print 'Is digit correct?, [y, n]'
                        dh, dw, dc = digit_loc.shape
                        jP = imagePrompt(digit_loc, dh, dw)
                        if jP.output == 'n':
                           roi = self.select_label(labeled_array, label_choice.next())
                           hog_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                           roi_hog_fd = hog(hog_roi, orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
                           guess = clf.predict(np.array([roi_hog_fd], dtype=float))[0]

                    elif iP.output == '.':
                        print 'Cycling next label'
                        roi = self.select_label(labeled_array, label_choice.next())
                        hog_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                        roi_hog_fd = hog(hog_roi, orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
                        guess = clf.predict(np.array([roi_hog_fd], dtype=float))[0]
                    elif iP.output == '!':
                        print 'Alerting possible mistake'
                        if i > 1:
                            mistakes.append(i-1)

                    elif iP.output == 'n':
                        input_number = iP.output
                    else:
                        numbers = ''.join(c for c in iP.output if c.isdigit())
                        if len(numbers):
                            input_number = numbers


                if iP.output == 'n':
                     print 'Target discarded!'
                else:
                    data.update({i:{
                        'x_org': x_org,
                        'y_org': y_org,
                        'input': input_number,
                        'roi': roi
                    }})

        check = False

        print 'Mistakes:'
        print mistakes
        print data.keys()
        while check is False:
            p = shell.Prompt('Check values?', ['y', 'n'])
            intval = False
            num = re.sub(r'[^\d]+', '', p.input)
            if num:
                intval = int(re.sub(r'[^\d]+', '', p.input))

            print intval
            if p.input == 'n':
                check = True
            elif intval in data.keys():
                d = data[intval]
                dh, dw = d['roi'].shape
                jP = imagePrompt(d['roi'], dh, dw, d['input'])
                data.update({intval:{
                    'x_org': d['x_org'],
                    'y_org': d['y_org'],
                    'input': int(jP.output),
                    'roi': d['roi']
                }})
            else:
                print 'Key not found'



        input_numbers = []
        for k, v in data.iteritems():
            self.point_ids.append(k)
            cv2.destroyAllWindows()
            self.number_input.update({k:v['input']})

            self.x_points.append(v['x_org'])
            self.y_points.append(v['y_org'])
            cv2.putText(im_check, str(v['input']), (v['x_org'], v['y_org']), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
            cv2.putText(im_blank, str(k), (v['x_org'], v['y_org']), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            cv2.putText(im_blank, str(k), (v['x_org'], v['y_org']), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            cv2.circle(im_points,(v['x_org'],v['y_org']), 7, (0,0,225), -1)
            self.taught_data.append(v['roi'])
            self.taught_labels.append(v['input'])
            input_numbers.append(v['input'])

        return im_points, im_check, im_blank, input_numbers

    def connect_digits(self, points, image):
        # Human test to ensure multi-character digits are grouped, then automatically combine based on location
        print 'Connecting digits'
        all_distances = []
        connections = []
        for i in range(len(points.T)):
            row = [points[0][i], points[1][i], points[2][i]]
            a = np.array((row[1], row[2], 1))
            distances = []
            counter_points = []
            for j in range(len(points.T)):
                row2 = [points[0][j], points[1][j], points[2][j]]
                if row2[0] != row[0]:
                    b = np.array((row2[1], row2[2], 1))
                    distances.append(np.linalg.norm(a-b))
                    counter_points.append(row2[0])

            # Nearest neighbour
            smallest_distance = min(distances)
            nn = counter_points[distances.index(smallest_distance)]
            all_distances.append(min(distances))
            pv = np.where(points[0]==nn)[0]

            # Check bearing isn't vertical or sub-vertical
            angle = abs(degrees(atan2(points[2][pv][0] - row[2], points[1][pv][0] - row[1])))
            angle_ranges = [(0,45), (135,225), (310,360)]
            if any(lower <= angle <= upper for (lower, upper) in angle_ranges):
                connections.append([row[1], row[2], points[1][pv][0], points[2][pv][0], smallest_distance, row[0], nn])

        distance_stdev = np.std(all_distances)
        distance_median = np.percentile(all_distances, 50)

        connection_threshold = distance_median + distance_stdev


        cT = connectionThresholder(connection_threshold, image, connections)
        cv2.destroyAllWindows()
        ok_connections = cT.current_connections
        groups = []

        for c in ok_connections:
            groups.append([c[5],c[6]])

        for g in groups:
            g = g.sort()

        # Remove duplicates
        groups = list(groups for groups,_ in itertools.groupby(groups))

        height, width, channels = image.shape
        connection_image = cT.connection_image

        p = shell.Prompt('Select individual connections?', ['y', 'n'])
        if p.input != 'n':
            while True:
                app = connectionSelect(connection_image)
                app.mainloop()
                roi = [int(x * sf) for x in app.dimensions]
                # cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 3)
                # height, width, channels = image.shape
                # im2 = cv2.resize(image, (int(width*sf2), int(height*sf2)))
                # cv2.imshow('test', im2)
                # cv2.waitKey(0)
                if len(roi) > 0:
                    points_in_roi = []
                    for i in range(len(points.T)):
                        if roiCheck(roi, [points[1][i], points[2][i]]) is True:
                            points_in_roi.append(points[0][i])

                    for pr in points_in_roi:
                        for g in groups:
                            for gi in g:
                                if gi == pr:
                                    g.remove(gi)

                    groups.append(points_in_roi)
                    im_copy = image.copy()

                    # Points in ROI
                    # Reset connections

                    for g in groups:

                        # Create new distance lists

                        if len(g) > 1:
                            # For each group id
                            p_points = []
                            c_points = []
                            for p in g:
                                # List through other ids
                                d = [] # Distances to original point
                                e = [] # Point ids
                                p_points.append(p)
                                gt = np.where(points[0]==p)[0]
                                a = np.array((points[1][gt], points[2][gt], 1))
                                for r in g:
                                    # If id is different
                                    if p != r:
                                        # Record distance
                                        pt = np.where(points[0]==r)[0]
                                        b = np.array((points[1][pt], points[2][pt], 1))
                                        d.append(np.linalg.norm(a-b))
                                        e.append(r)

                                nn = e[d.index(min(d))]
                                c_points.append(nn)

                            for m in range(len(c_points)):
                                c1p = np.where(points[0]==p_points[m])[0]
                                c2p = np.where(points[0]==c_points[m])[0]
                                c1 = (points[1][c1p], points[2][c1p])
                                c2 = (points[1][c2p], points[2][c2p])
                                cv2.line(im_copy,c1,c2,(255,0,0),3)

                    connection_image = cv2.resize(im_copy, (int(width*sf2), int(height*sf2)))

                p = shell.Prompt('Select more regions?', ['y', 'n'])
                if p.input == 'n':
                    break

        a = np.array(groups)
        a = np.hstack(a)
        connected_points = np.unique(a)

        singular_digits = np.setdiff1d(points[0], connected_points)
        connected_groups = []

        # Group interconnected
        for g in groups:
            c = [g]
            for k in range(len(groups)):
                if not len(g) == len(np.setdiff1d(g, groups[k])):
                    c.append(groups[k])
                    groups[k] = []

            connected_groups.append(c)

        for m in range(len(connected_groups)):
            b = np.array(connected_groups[m])
            b = np.hstack(b)
            connected_groups[m] = np.unique(b).tolist()


        # Remove empties
        connected_groups = [x for x in connected_groups if x]

        numlist = []

        for s in singular_digits:
            numlist.append(int(self.number_input[s]))

        for g in connected_groups:
            num_string = ''
            # Reorder according to left-right position
            x_coords = []
            print g
            for i in g:
                print i
                pv = np.where(points[0]==i)[0]
                # i
                if pv:
                    x_coords.append([i, points[1][pv][0]])

            g_x_coords = np.array(x_coords)
            l = g_x_coords[np.argsort(g_x_coords[:, 1])]
            for n in l:
                input_string = str(self.number_input[int(n[0])])
                num_string += input_string
            numlist.append(num_string)

        final_numbers = map(int, numlist)

        return final_numbers

    def addFolder(self, parent, name):
        name_check = False
        name = str(name)
        orig_name = name
        while name_check is False:
            folder_path = os.path.join(parent, name)
            if os.path.isdir(folder_path):
                randfive = ''.join(random.choice('0123456789ABCDEF') for i in range(5))
                name = ''.join([orig_name, '_', randfive])
            else:
                name_check = True

        os.makedirs(folder_path)

        return folder_path

    def save_data(self, numbers, image_choice, im_check, im_blank, input_numbers):

        # Save data to directory
        meta = self.meta
        self.meta.update({'Image': os.path.basename(image_choice)})
        root_path = self.output_path

        fan_path = os.path.join(root_path, meta['fan'])
        if not os.path.isdir(fan_path):
            os.makedirs(fan_path)

        surface_path = os.path.join(fan_path, meta['surface'])
        if not os.path.isdir(surface_path):
            os.makedirs(surface_path)

        data_path = os.path.join(surface_path, meta['site'])

        if os.path.exists(data_path):
            # do we want to replace?
            meta_path = os.path.join(data_path, str(meta['name'])+'_meta.yml')
            if os.path.exists(meta_path):
                data_path = self.addFolder(surface_path, meta['site'])
        else:
             data_path = self.addFolder(surface_path, meta['site'])

        print 'Save directory: '+data_path
        meta_path = os.path.join(data_path, str(meta['name'])+'_meta.yml')

        # Meta yaml
        print 'Saving meta'
        ruamel.yaml.dump(meta, open(meta_path, 'w'), Dumper=ruamel.yaml.RoundTripDumper)

        # Data CSV
        prefix = str(meta['name'])+'_'
        csv_name = prefix+'wolman.csv'
        csv_path = os.path.join(data_path, csv_name)

        print 'Saving csv: '+csv_path

        with open(csv_path, 'wb') as csvfile:
            wr = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for n in numbers:
                wr.writerow([n])

        csv_name = prefix+'input.csv'
        csv_path = os.path.join(data_path, csv_name)

        print 'Saving data input copy: '+csv_path

        with open(csv_path, 'wb') as csvfile:
            wr = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for iN in input_numbers:
                wr.writerow([iN])

        # Save taught data
        print 'Saving taught data'
        t_data_name = prefix+'learn_data.npy'
        t_label_name = prefix+'learn_labels.npy'
        np.save(os.path.join(data_path,t_data_name), self.taught_data)
        np.save(os.path.join(data_path,t_label_name), self.taught_labels)

        # Save digit image
        print 'Saving images'
        cv2.imwrite(os.path.join(data_path,prefix+'digits.png'), im_check)
        cv2.imwrite(os.path.join(data_path,prefix+'roi.png'), im_blank)


        # Copy image
        image_basename = os.path.basename(image_choice)
        if not os.path.isfile(os.path.join(data_path, image_basename)):
            copyfile(image_choice, os.path.join(data_path, image_basename))


shutdown_event = threading.Event()

class imagePrompt():
    def __init__(self, image, h, w, guess=False):
        self.root = Tk()
        self.output = False
        self.guess = False
        if guess is not False:
            self.guess = True

        self.entry_text = StringVar(self.root, value=guess)
        self.entry = Entry(self.root, textvariable=self.entry_text)
        self.entry.pack()
        self.entry.focus_set()
        self.canvas = Canvas(self.root, width=w, height=h)
        self.canvas.pack()
        im = PIL.Image.fromarray(image)
        tk_img = ImageTk.PhotoImage(im)
        self.canvas.create_image(0, 0, image=tk_img, anchor='nw')
        self.entry.bind('<Key-Return>', self.response)
        self.entry.bind('<Key>', self.remove_guess)
        self.root.mainloop()

    def response(self, event):
        self.output = self.entry.get()
        self.root.destroy()

    def remove_guess(self, event):
        if self.guess is True:
            self.entry.config(textvariable=StringVar(self.root, value=''))
            self.guess = False

def roiCheck(roi, point):
    inside = False
    x_inside = False
    y_inside = False

    x_lims = [roi[0], roi[2]]
    x_lims.sort()
    y_lims = [roi[1], roi[3]]
    y_lims.sort()
    if x_lims[0] < point[0] < x_lims[1]:
        x_inside = True

    if y_lims[0] < point[1] < y_lims[1]:
        y_inside = True

    if x_inside and y_inside:
        inside = True

    return inside

class connectionThresholder():

    def __init__(self, starting_val, image, connections):
        self.root = Tk()
        self.num = False
        self.threshold = starting_val
        self.connections = connections
        Label(self.root, text="Threshold").grid(row=0)
        v = StringVar(self.root, value=starting_val)
        self.el = Entry(self.root, textvariable=v)
        self.set_btn = Button(self.root, text="Set", command = self.set_response)
        self.ok_btn = Button(self.root, text="OK", command = self.ok_response)
        self.el.grid(row=0, column=1)
        self.set_btn.grid(row=1, column=1)
        self.ok_btn.grid(row=2, column=1)
        self.el.bind("<Return>", self.set_response)
        self.current_connections = connections
        self.original_image = image
        self.connection_image = None
        self.load_window()
        self.el.focus()
        self.root.mainloop()

    def load_window(self):
        im = self.original_image.copy()
        self.current_connections = []
        for c in self.connections:
            if c[4] < self.threshold:
                self.current_connections.append(c)
                cv2.line(im,(c[0],c[1]),(c[2],c[3]),(255,0,0),3)

        height, width, channels = im.shape
        im = cv2.resize(im, (int(width*sf2), int(height*sf2)))
        self.connection_image = im
        self.current_image = cv2.imshow('image', im)
        cv2.waitKey(100)

    def get_num(self):
        return self.num

    def set_response(self):
        self.threshold = float(self.el.get())
        self.load_window()

    def ok_response(self):
        self.num = self.el.get()
        self.imnum = self.el.get()
        self.root.destroy()
        cv2.destroyAllWindows()

class connectionSelect(Tk):
    def __init__(self, image):
        Tk.__init__(self)
        self.x = self.y = 0
        self.im = image
        dh, dw, dc = image.shape
        self.canvas = Canvas(self, width=dw, height=dh, cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.dimensions = []
        self._draw_image()

    def _draw_image(self):
        im = PIL.Image.fromarray(self.im)
        self.tk_im = ImageTk.PhotoImage(image=im)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # create rectangle if not yet exist
        #if not self.rect:
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1)

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
        self.dimensions = [self.start_x, self.start_y, curX, curY]

    def on_button_release(self, event):
        self.destroy()


sf = 2.5
sf2 = 1/float(sf)

clf = joblib.load('digit_learn_all.pkl')



app.setup()

app.run()


# Using my own config for now
sam_config_path = '/home/sb708/.sam'
config_raw = ruamel.yaml.load(open(sam_config_path), ruamel.yaml.RoundTripLoader)
config = {
    'image_directory': None,
    'output_directory': None
}

if config_raw.has_key('digit_recogniser'):
    config = config_raw['digit_recogniser']

cont = True
first_run = True
while cont is True:
    if first_run is not True:
        p = shell.Prompt("Transcribe new image?", ['y','n'])
        if p.input is 'y':
            print("Running...")
            tsc = Transcriber(config)
        else:
                cont = False
    else:
        first_run = False
        print("Running...")
        tsc = Transcriber(config)

print 'Finishing up...'
app.close()