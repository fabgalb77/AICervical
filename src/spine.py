import numpy as np
import cv2
from copy import deepcopy
import math
import pandas as pd

class spine:
    def __init__(self, c, ps):
        self.images = None
        self.pixel_size = ps
        self.coords = np.zeros((29, 2), dtype= int)
        ipoint = 0
        for x, y in c:
            self.coords[ipoint, 0] = int(x)
            self.coords[ipoint, 1] = int(y)
            ipoint += 1
        return

    def get_parameters(self):
        data = None
        t1_slope = round(self.t1_slope(), 1)
        c7_slope = round(self.c7_slope(), 1)
        sva_px, sva_mm = self.sva()
        cranial_tilt = round(self.cranialTilt(), 1)
        cervical_tilt = round(self.cervicalTilt(), 1)
        c0c2 = round(self.c0c2(), 1)
        c2c6 = round(self.c2c6(), 1)
        c2c7 = round(self.c2c7(), 1)
        rd_px, rd_mm = self.redlund()
        if self.pixel_size > 0.:
            sva_mm = round(sva_mm, 1)
            rd_mm = round(rd_mm, 1)
            data = {'parameter' : ['T1 slope', 'C7 slope', 'SVA', 'cranial tilt', 'cervical tilt', 'C0-C2', 'C2-C6', 'C2-C7', 'Redlund-Johnell dist.'],
                    'value' : [t1_slope, c7_slope, sva_mm, cranial_tilt, cervical_tilt, c0c2, c2c6, c2c7, rd_mm],
                    'unit' : ['deg', 'deg', 'mm', 'deg', 'deg', 'deg', 'deg', 'deg', 'mm']}
        else:
            sva_px = round(sva_px, 0)
            rd_px = round(rd_px, 0)
            data = {'parameter' : ['T1 slope', 'C7 slope', 'SVA', 'cranial tilt', 'cervical tilt', 'C0-C2', 'C2-C6', 'C2-C7', 'Redlund-Johnell dist.'],
                    'value' : [t1_slope, c7_slope, sva_px, cranial_tilt, cervical_tilt, c0c2, c2c6, c2c7, rd_px],
                    'unit' : ['deg', 'deg', 'pixels', 'deg', 'deg', 'deg', 'deg', 'deg', 'pixels']}
        df = pd.DataFrame(data)
        return df

    def get_direction(self):
        votes_left = 0 
        votes_right = 0
        for iv in range(6):
            p1 = iv * 4 + 2
            p2 = iv * 4 + 3
            p3 = iv * 4 + 4
            p4 = iv * 4 + 5
            if self.coords[p2, 0] > self.coords[p1, 0]:
                votes_right += 1
            else:
                votes_left += 1
            if self.coords[p3, 0] > self.coords[p4, 0]:
                votes_right += 1
            else:
                votes_left += 1
        if votes_right > votes_left:
            return 'right'
        else:
            return 'left'

    def plot_all(self, img):
        img_dc = deepcopy(img)
        img_dc = cv2.line(img_dc, (self.coords[0,0], self.coords[0,1]), (self.coords[1,0], self.coords[1,1]), (0, 255, 0), 3) 
        for iv in range(6):
            p1 = iv * 4 + 2
            p2 = iv * 4 + 3
            p3 = iv * 4 + 4
            p4 = iv * 4 + 5
            img_dc = cv2.line(img_dc, (self.coords[p1,0], self.coords[p1,1]), (self.coords[p2,0], self.coords[p2,1]), (0, 255, 0), 3) 
            img_dc = cv2.line(img_dc, (self.coords[p2,0], self.coords[p2,1]), (self.coords[p3,0], self.coords[p3,1]), (0, 255, 0), 3) 
            img_dc = cv2.line(img_dc, (self.coords[p3,0], self.coords[p3,1]), (self.coords[p4,0], self.coords[p4,1]), (0, 255, 0), 3) 
            img_dc = cv2.line(img_dc, (self.coords[p4,0], self.coords[p4,1]), (self.coords[p1,0], self.coords[p1,1]), (0, 255, 0), 3) 

        img_dc = cv2.line(img_dc, (self.coords[22,0], self.coords[22,1]), (self.coords[28,0], self.coords[28,1]), (127, 255, 0), 3) 
        img_dc = cv2.line(img_dc, (self.coords[23,0], self.coords[23,1]), (self.coords[28,0], self.coords[28,1]), (127, 255, 0), 3) 
        img_dc = cv2.line(img_dc, (self.coords[26,0], self.coords[26,1]), (self.coords[27,0], self.coords[27,1]), (255, 0, 0), 3) 

        return img_dc

    def plot_angle(self, img, points, text):
        img_dc = deepcopy(img)
        p1l1 = self.get_point(points[0][0])
        p2l1 = self.get_point(points[0][1])
        p1l2 = self.get_point(points[1][0])
        p2l2 = self.get_point(points[1][1])
        img_dc = cv2.line(img_dc, (p1l1[0], p1l1[1]), (p2l1[0], p2l1[1]), (0, 255, 0), 3) 
        img_dc = cv2.line(img_dc, (p1l2[0], p1l2[1]), (p2l2[0], p2l2[1]), (0, 255, 0), 3) 
        p_text = ( int((p1l1[0] + p2l1[0] + p1l2[0] + p2l2[0]) / 4), int((p1l1[1] + p2l1[1] + p1l2[1] + p2l2[1]) / 4))
        text_thk = 2 
        if img_dc.shape[0] < 1000:
            text_thk = 1
        img_dc = cv2.putText(img_dc, text, p_text, cv2.FONT_HERSHEY_SIMPLEX,  1.3 * (img_dc.shape[0] / 2500.), (255, 0, 0), text_thk, cv2.LINE_AA)
        return img_dc

    def plot_angle_horizontal(self, img, points, text):
        img_dc = deepcopy(img)
        p1l1 = self.get_point(points[0])
        p2l1 = self.get_point(points[1])
        p1l2 = self.get_point(points[1])
        p2l2 = (self.coords[points[0], 0], self.coords[points[1], 1])
        img_dc = cv2.line(img_dc, (p1l1[0], p1l1[1]), (p2l1[0], p2l1[1]), (0, 255, 0), 3) 
        img_dc = cv2.line(img_dc, (p1l2[0], p1l2[1]), (p2l2[0], p2l2[1]), (0, 255, 0), 3) 
        p_text = ( int((p1l1[0] + p2l1[0] + p1l2[0] + p2l2[0]) / 4), int((p1l1[1] + p2l1[1] + p1l2[1] + p2l2[1]) / 4 - img_dc.shape[0] * 0.03) )
        text_thk = 2 
        if img_dc.shape[0] < 1000:
            text_thk = 1
        img_dc = cv2.putText(img_dc, text, p_text, cv2.FONT_HERSHEY_SIMPLEX,  1.3 * (img_dc.shape[0] / 2500.), (255, 0, 0), text_thk, cv2.LINE_AA)
        return img_dc

    def plot_cranial_tilt(self, img):
        img_dc = deepcopy(img)
        p1l1 = self.get_midT1()
        p2l1 = self.get_midC2()
        p1l2 = self.get_midC2()
        p2l2 = (p1l2[0], p1l1[1])
        img_dc = cv2.line(img_dc, (p1l1[0], p1l1[1]), (p2l1[0], p2l1[1]), (0, 255, 0), 3) 
        img_dc = cv2.line(img_dc, (p1l2[0], p1l2[1]), (p2l2[0], p2l2[1]), (0, 255, 0), 3) 
        p_text = ( int((p1l1[0] + p2l1[0] + p1l2[0] + p2l2[0]) / 4), int((p1l1[1] + p2l1[1] + p1l2[1] + p2l2[1]) / 4 - img_dc.shape[0] * 0.03) )
        text = 'cranial tilt: {:.1f} deg'.format(self.cranialTilt())
        text_thk = 2 
        if img_dc.shape[0] < 1000:
            text_thk = 1
        img_dc = cv2.putText(img_dc, text, p_text, cv2.FONT_HERSHEY_SIMPLEX,  1.3 * (img_dc.shape[0] / 2500.), (255, 0, 0), text_thk, cv2.LINE_AA)
        return img_dc

    def plot_cervical_tilt(self, img):
        img_dc = deepcopy(img)
        p1l1 = self.get_midT1()
        p2l1 = self.get_midC2()
        p1l2 = self.get_point(0)
        p2l2 = self.get_point(1)
        p1l3 = self.get_midT1()
        p2l3_vx = self.coords[0, 1] - self.coords[1, 1]
        p2l3_vy = -(self.coords[0, 0] - self.coords[1, 0])
        p2l3 = (p1l3[0] + p2l3_vx, p1l3[1] + p2l3_vy)
        p1l4 = self.get_midT1()
        p2l4 = (p1l3[0] - p2l3_vx, p1l3[1] - p2l3_vy)
        img_dc = cv2.line(img_dc, (p1l1[0], p1l1[1]), (p2l1[0], p2l1[1]), (0, 255, 0), 3) 
        img_dc = cv2.line(img_dc, (p1l2[0], p1l2[1]), (p2l2[0], p2l2[1]), (0, 0, 255), 2) 
        if self.get_direction() == 'left':
            img_dc = cv2.line(img_dc, (p1l3[0], p1l3[1]), (p2l3[0], p2l3[1]), (0, 255, 0), 3) 
        else:
            img_dc = cv2.line(img_dc, (p1l4[0], p1l4[1]), (p2l4[0], p2l4[1]), (0, 255, 0), 3) 
        p_text = ( int((p1l1[0] + p2l1[0] + p1l2[0] + p2l2[0]) / 4), int((p1l1[1] + p2l1[1] + p1l2[1] + p2l2[1]) / 4 - img_dc.shape[0] * 0.03) )
        text = 'cervical tilt: {:.1f} deg'.format(self.cervicalTilt())
        text_thk = 2 
        if img_dc.shape[0] < 1000:
            text_thk = 1
        img_dc = cv2.putText(img_dc, text, p_text, cv2.FONT_HERSHEY_SIMPLEX,  1.3 * (img_dc.shape[0] / 2500.), (255, 0, 0), text_thk, cv2.LINE_AA)
        return img_dc

    def plot_sva(self, img):
        img_dc = deepcopy(img)
        p1l1 = self.get_midC2()
        p2l2 = self.get_point(2)
        p1l2 = (p1l1[0], p2l2[1])
        p2l1 = (p1l1[0], p2l2[1])
        
        print(p1l1)
        print(p2l1)
        print(p1l2)
        print(p2l2)

        img_dc = cv2.line(img_dc, (p1l1[0], p1l1[1]), (p2l1[0], p2l1[1]), (0, 255, 0), 3) 
        img_dc = cv2.line(img_dc, (p1l2[0], p1l2[1]), (p2l2[0], p2l2[1]), (0, 255, 0), 3) 
        p_text = ( int((p1l1[0] + p2l1[0] + p1l2[0] + p2l2[0]) / 4), int((p1l1[1] + p2l1[1] + p1l2[1] + p2l2[1]) / 4 - img_dc.shape[0] * 0.03) )
        sva_px, sva_mm = self.sva()
        if sva_mm > 0:
            text = 'SVA : {:.1f} mm'.format(sva_mm)
        else:
            text = 'SVA : {} px'.format(int(sva_px))
        text_thk = 2 
        if img_dc.shape[0] < 1000:
            text_thk = 1
        img_dc = cv2.putText(img_dc, text, p_text, cv2.FONT_HERSHEY_SIMPLEX,  1.3 * (img_dc.shape[0] / 2500.), (255, 0, 0), text_thk, cv2.LINE_AA)
        return img_dc

    def plot_redlund(self, img):
        def get_closest_point(a, b, p):
            a_to_p = (p[0] - a[0], p[1] - a[1])     
            a_to_b = (b[0] - a[0], b[1] - a[1])

            atb2 = a_to_b[0]**2 + a_to_b[1]**2
            atp_dot_atb = a_to_p[0]*a_to_b[0] + a_to_p[1]*a_to_b[1]
                                            
            t = atp_dot_atb / atb2          
            return (int(a[0] + a_to_b[0]*t), int(a[1] + a_to_b[1]*t) )

        img_dc = deepcopy(img)
        p1l1 = self.get_point(26)
        p2l1 = self.get_point(27)
        p1l2 = self.get_point(24)
        p2l2 = self.get_point(25)
        p1l3 = (int((self.coords[24, 0] + self.coords[25, 0]) / 2), int((self.coords[24, 1] + self.coords[25, 1]) / 2) )
        p2l3 = get_closest_point(p1l1, p2l1, p1l3)

        img_dc = cv2.line(img_dc, (p1l1[0], p1l1[1]), (p2l1[0], p2l1[1]), (0, 0, 255), 3) 
        img_dc = cv2.line(img_dc, (p1l2[0], p1l2[1]), (p2l2[0], p2l2[1]), (0, 0, 255), 3) 
        img_dc = cv2.line(img_dc, (p1l3[0], p1l3[1]), (p2l3[0], p2l3[1]), (0, 255, 0), 3) 
        p_text = ( int((p1l1[0] + p2l1[0] + p1l2[0] + p2l2[0]) / 4), int((p1l1[1] + p2l1[1] + p1l2[1] + p2l2[1]) / 4) )
        rd_px, rd_mm = self.redlund()
        if rd_mm > 0:
            text = 'Redlund-Johnell : {:.1f} mm'.format(rd_mm)
        else:
            text = 'Redlund-Johnell : {} px'.format(int(rd_px))
        text_thk = 2 
        if img_dc.shape[0] < 1000:
            text_thk = 1
        img_dc = cv2.putText(img_dc, text, p_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3 * (img_dc.shape[0] / 2500.), (255, 0, 0), text_thk, cv2.LINE_AA)
        return img_dc
        

    def get_point(self, n):
        return (self.coords[n, 0], self.coords[n, 1])

    def calculate_m(self, points):
        p1 = self.get_point(points[0])
        p2 = self.get_point(points[1])
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        return m

    def t1_slope(self):
        m = self.calculate_m((0, 1))
        alpha = np.arctan(m)
        if self.get_direction() == 'left':
            alpha = -alpha
        return alpha * 180 / np.pi

    def c2c7(self):
        m1 = self.calculate_m((24, 25))
        m2 = self.calculate_m((5, 4))
        angle_1 = np.arctan(m1)
        angle_2 = np.arctan(m2)

        if self.get_direction() == 'left':
            angle_1 = -angle_1
            angle_2 = -angle_2
        return (angle_2 - angle_1) * 180 / np.pi

    def get_midT1(self):
        return (int((self.coords[0, 0] + self.coords[1, 0]) / 2), int((self.coords[0, 1] + self.coords[1, 1]) / 2) )

    def get_midC2(self):
        return (int((self.coords[22, 0] + self.coords[23, 0] + self.coords[24, 0] + self.coords[25, 0]) / 4), int((self.coords[22, 1] + self.coords[23, 1] + self.coords[24, 1] + self.coords[25, 1]) / 4) )

    def cranialTilt(self):
        p1 = self.get_midT1()
        p2 = self.get_midC2()

        #m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        #alpha = np.arctan(m)
        alpha = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        alpha = 90 + alpha * 180 / np.pi

        if self.get_direction() == 'left':
            alpha = -alpha
        return alpha

    def cervicalTilt(self):
        p1 = self.get_midT1()
        p2 = self.get_midC2()
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])

        p1p3_vx = self.coords[0, 1] - self.coords[1, 1]
        p1p3_vy = -(self.coords[0, 0] - self.coords[1, 0])
        p3 = (p1[0] + p1p3_vx, p1[1] + p1p3_vy)
        if self.get_direction() == 'right':
            p3 = (p1[0] - p1p3_vx, p1[1] - p1p3_vy)

        angle = math.atan2(p3[1] - p1[1], p3[0] - p1[0]) - math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        
        if self.get_direction() == 'left':
            angle = -angle
    
        return angle * 180. / math.pi 

    def sva(self):
        p1 = self.get_midC2()
        p2 = self.get_point(2)

        sva = p1[0] - p2[0]

        if self.get_direction() == 'left':
            sva = -sva

        if self.pixel_size > 0.:
            return sva, sva * self.pixel_size
        else:
            return sva, -1.

    def c0c2(self):
        m1 = self.calculate_m((26, 27))
        m2 = self.calculate_m((25, 24))
        angle_1 = np.arctan(m1)
        angle_2 = np.arctan(m2)

        if self.get_direction() == 'left':
            angle_1 = -angle_1
            angle_2 = -angle_2
        return (angle_2 - angle_1) * 180 / np.pi
        
    def c2c6(self):
        m1 = self.calculate_m((24, 25))
        m2 = self.calculate_m((9, 8))
        angle_1 = np.arctan(m1)
        angle_2 = np.arctan(m2)

        if self.get_direction() == 'left':
            angle_1 = -angle_1
            angle_2 = -angle_2
        return (angle_2 - angle_1) * 180 / np.pi

    def c7_slope(self):
        m = self.calculate_m((2, 3))
        alpha = np.arctan(m)
        if self.get_direction() == 'left':
            alpha = -alpha
        return alpha * 180 / np.pi

    def redlund(self):
        p0 = ((self.coords[24, 0] + self.coords[25, 0]) / 2, (self.coords[24, 1] + self.coords[25, 1]) / 2)
        p1 = self.get_point(26)
        p2 = self.get_point(27)

        num = np.fabs(((p2[0] - p1[0]) * (p1[1] - p0[1])) - ((p1[0] - p0[0]) * (p2[1] - p1[1])))
        den = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        d = num / den

        if self.pixel_size > 0.:
            return d, d * self.pixel_size
        else:
            return d, -1.