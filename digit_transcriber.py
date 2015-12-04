import cv2
import scipy
import PIL.Image
import os
import time
from Tkinter import *
from scipy import ndimage as nd
from math import degrees, atan2
import csv
import ruamel.yaml
import numpy as np
import threading
from shutil import copyfile
import random
import itertools

from cement.core import foundation, controller
from cement.core.controller import expose
from cement.utils import shell


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

        self.taught_data = np.empty([1])
        self.taught_labels = np.empty([1])
        self.x_points = []
        self.y_points = []
        self.point_ids = []
        self.number_input = {}
        self.check_workspace(config)
        self.get_meta()

        rects, im_blank, im, im_th, image_choice, im_check = self.load_image()

        self.transcribe(rects, im, im_th, im_blank, im_check)

        points = np.array([self.point_ids, self.x_points, self.y_points])

        numbers = self.connect_digits(points, im_blank)
        self.save_data(numbers, image_choice, im_check)

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

        # Save config
        config = {
            'digit_recogniser': {
                'image_directory': self.images_path,
                'output_directory': self.output_path
            }
        }
        ruamel.yaml.dump(config, open(sam_config_path, 'w'), Dumper=ruamel.yaml.RoundTripDumper)

    def get_meta(self):
        # Context and data labels
        fan = None
        surface = None
        site = None
        name = None
        cover = None
        location = None

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
        image_choice = 'photo_5.jpg'
        while image_choice is None:
            p = shell.Prompt("Image: ")
            print 'Checking '+self.images_path
            if os.path.exists(p.input):
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

        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        return rects, im_blank, im, im_th, image_choice, im_check

    def transcribe(self, rects, im, im_th, im_blank, im_check):
        # Use human to tell us what digits are

        i = 0
        im_orig = im.copy()
        for rect in rects:
            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

            i = i +1
            if i > 20:
                break

            if roi.size:

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
                mne = np.ma.masked_not_equal(labeled_array, target_label)
                inverted = np.invert(mne.mask)
                roi = (inverted * 255).astype(np.uint8)

                x_org = rect[0] + (rect[2]/2)
                y_org = rect[1] + (rect[3]/2)

                digit_loc = im_orig.copy()
                cv2.rectangle(digit_loc, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
                dh, dw, dc = digit_loc.shape
                #roi = cv2.resize(roi, (100, 100))
                digit_loc = cv2.resize(digit_loc, (int(dw*0.5), int(dh*0.5)))



                input_number = False

                while input_number is False:
                    c = threading.Thread(name='check', args=([digit_loc]), target=show_image)
                    d = threading.Thread(name='image', args=([roi]), target=show_image)
                    d.start()
                    p = shell.Prompt("Type digit you can see: ")
                    shutdown_event.set()
                    cv2.destroyAllWindows()
                    print p.input
                    if p.input == 'check':
                        c.start()
                        shell.Prompt("Press enter to continue", default='ENTER')
                        shutdown_event.set()
                        cv2.destroyAllWindows()
                    else:
                        input_number = p.input


                if p.input is not 'n':
                    self.point_ids.append(i)

                    cv2.destroyAllWindows()

                    self.number_input.update({i:input_number})
                    #self.number_input.update({i:random.randint(0,9)})


                    self.x_points.append(x_org)
                    self.y_points.append(y_org)
                    cv2.putText(im_check, str(input_number), (x_org, y_org), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
                    cv2.circle(im_blank,(x_org,y_org), 5, (0,0,225), -1)
                    np.append(self.taught_data, roi)
                    np.append(self.taught_labels, input_number)
                else:
                    print 'Target discarded!'

    def connect_digits(self, points, im_blank):
        # Human test to ensure multi-character digits are grouped, then automatically combine based on location

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

        cT = connectionThresholder(connection_threshold, im_blank, connections)
        set_threshold = cT.get_num()
        ok_connections = cT.current_connections
        groups = []

        for c in ok_connections:
            groups.append([c[5],c[6]])

        for g in groups:
            g = g.sort()

        # Remove duplicates
        groups = list(groups for groups,_ in itertools.groupby(groups))

        a = np.array(groups)
        a.flatten()
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
            b.flatten()
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
            for i in g:
                x_coords.append([i, self.x_points[i-1]])

            g_x_coords = np.array(x_coords)
            l = g_x_coords[np.argsort(g_x_coords[:, 1])]
            for n in l:
                num_string = num_string+str(self.number_input[int(n[0])])

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

    def save_data(self, numbers, image_choice, im_check):

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

        meta_path = os.path.join(data_path, str(meta['name'])+'_meta.yml')

        # Meta yaml
        ruamel.yaml.dump(meta, open(meta_path, 'w'), Dumper=ruamel.yaml.RoundTripDumper)

        # Data CSV
        prefix = str(meta['name'])+'_'
        csv_name = prefix+'wolman.csv'
        csv_path = os.path.join(data_path, csv_name)

        with open(csv_path, 'wb') as csvfile:
            wr = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for n in numbers:
                wr.writerow([n])

        # Save taught data
        t_data_name = prefix+'learn_data.npy'
        t_label_name = prefix+'learn_labels.npy'
        np.save(os.path.join(data_path,t_data_name), self.taught_data)
        np.save(os.path.join(data_path,t_label_name), self.taught_labels)

        # Save digit image
        cv2.imwrite(os.path.join(data_path,prefix+'digits.png'), im_check)

        # Copy image
        image_basename = os.path.basename(image_choice)
        if not os.path.isfile(os.path.join(data_path, image_basename)):
            copyfile(image_choice, os.path.join(data_path, image_basename))


shutdown_event = threading.Event()

def show_image(photo):
    cv2.imshow('image', photo)
    cv2.waitKey(0)
    while not shutdown_event.is_set():
        time.sleep(1.0)

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
        self.load_window()
        self.el.focus()
        self.root.mainloop()

    def load_window(self):
        im = self.original_image.copy()
        self.current_connections = []
        for c in self.connections:
            if c[4] < self.threshold:
                self.current_connections.append(c)
                cv2.line(im,(c[0],c[1]),(c[2],c[3]),(255,0,0),1)

        height, width, channels = im.shape
        im = cv2.resize(im, (int(width*0.5), int(height*0.5)))

        self.current_image = cv2.imshow('image', im)
        cv2.waitKey(100)

    def get_num(self):
        return self.num

    def set_response(self):
        self.threshold = float(self.el.get())
        self.load_window()

    def ok_response(self):
        self.num = self.el.get()
        self.root.destroy()
        cv2.destroyAllWindows()

try:
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

finally:
    print 'Finishing up...'
    app.close()