import numpy as np
import struct
from help_func.logging import LoggingHelper
from help_func.help_python import myUtil
import copy

class LearningIndex:
    TEST = 0
    TRAINING = 1
    VALIDATION = 2
    MAX_NUM_COMPONENT = 3
    INDEX_DIC = {TEST:"TEST", TRAINING:"TRAINING", VALIDATION:"VALIDATION"}

class Component:
    COMPONENT_Y = 0
    COMPONENT_Cb = 1
    COMPONENT_Cr = 2
    MAX_NUM_COMPONENT = 3


class ChromaFormat:
    YCbCr4_0_0 = 0
    YCbCr4_2_0 = 1
    YCbCr4_4_4 = 2
    MAX_NUM_COMPONENT = 3

class PictureFormat:
    ORIGINAL = 0
    PREDICTION = 1
    RECONSTRUCTION = 2
    UNFILTEREDRECON = 3
    MAX_NUM_COMPONENT = 4

class TuList:
    # 0: width, 1: height, 2: x_pos 3: y_pos, 4 : qp, 5 : mode ..

    def __init__(self, tulist):
        if tulist is None:
            self.tulist = np.array([[],[],[],[]], dtype='int32')
        else:
            self.tulist = tulist
        # Only Use +=, -=, *=, /=, //= or Reference...
        self.width = self.tulist[0]
        self.height = self.tulist[1]
        self.x_pos = self.tulist[2]
        self.y_pos = self.tulist[3]


    def resetMember(self):
        self.width = self.tulist[0]
        self.height = self.tulist[1]
        self.x_pos = self.tulist[2]
        self.y_pos = self.tulist[3]

    def getwidth(self):
        return self.tulist[0]

    def getheight(self):
        return self.tulist[1]
    def getx_pos(self):
        return self.tulist[2]
    def gety_pos(self):
        return self.tulist[3]

    def tuNum(self):
        return len(self.tulist[0])

    def dataShape(self):
        return np.shape(self.tulist)


    # 0: width, 1: height, 2: x_pos 3: y_pos, 4 : qp, 5 : mode ..
    def containTuList(self, area):
        filtered = self.tulist[:,~np.any([(self.tulist[1] + self.tulist[3]) < area.y,
                                          (self.tulist[2] + self.tulist[0]) < area.x,
                                           self.tulist[3] > (area.y + area.height),
                                           self.tulist[2] > (area.x + area.width)
                                           ], axis = 0)]
        filtered[2, filtered[2]<area.x] = area.x
        filtered[3, filtered[3]<area.y] = area.y
        # if filtered.min() < 0:
        #     pass
        filtered[0, (filtered[0] + filtered[2]) > (area.x + area.width)] =\
            (area.width + area.x) - filtered[2, (filtered[0] + filtered[2]) > (area.x + area.width)]
        filtered[1, (filtered[1] + filtered[3]) > (area.y + area.height)] -= \
            (area.height + area.y) - filtered[3, (filtered[1] + filtered[3]) > (area.y + area.height)]
        np.abs(filtered, out=filtered) # abs inplace
        return filtered



    def relateTo(self, pos):
        self.tulist[:, 2] = self.tulist[:, 2] - pos.x
        self.tulist[:, 3] = self.tulist[:, 3] - pos.y
        return

    def saveTuList(self, f):
        if not len(self.tulist[0]):
            f.write(struct.pack('<2i', *[0,0]))
            LoggingHelper.get_instance().logger.error('No Tu info')
            return False
        self.tulist[2] -= self.tulist[2].min()
        self.tulist[3] -= self.tulist[3].min()
        row, num = self.dataShape()
        rowandnum = np.array((row, num), dtype= 'int32')
        data = self.tulist.flatten()
        f.write(struct.pack('<2i', *rowandnum))
        f.write(struct.pack('<' + str(len(data)) + 'h', *data))
        return True



    @staticmethod
    def loadTuList_Old(f):
        row, num = struct.unpack('<2h', f.read(4))
        tulist = np.array(struct.unpack('<' + str(row * num) + 'h',
                                        f.read(2*row*num)), dtype = 'int16').reshape((row, num))

        return TuList(tulist)

    @staticmethod
    def loadTuList(f):
        row, num = struct.unpack('<2i', f.read(8))
        if not row:
            return TuList(None)
        tulist = np.array(struct.unpack('<' + str(row * num) + 'h',
                                        f.read(2*row*num)), dtype = 'int16').reshape((row, num))

        return TuList(tulist)

    def getTuMaskFromIndex(self, idx, height, width):
        # 0: width, 1: height, 2: x_pos 3: y_pos, 4 : qp, 5 : mode
        tumap = np.zeros((height, width))
        for i in range(len(self.width)):
            tumap[self.y_pos[i]:self.y_pos[i] + self.height[i] ,
            self.x_pos[i] : self.x_pos[i] + self.width[i]] = self.tulist[4+idx, i]
        return tumap

    def getMeanTuValue(self, idx):
        return np.mean(self.tulist[4+idx])

    def getTUMaskFromIndex_OneHot(self, idx, height, width, rangelist):
        tumap = np.zeros((len(rangelist), height, width), dtype='float32')
        for i in range(len(self.width)):
            tuvalue = self.tulist[4 + idx, i]
            assert tuvalue in rangelist, "tulist must in rangelist [%s]" %tuvalue
            tumap[rangelist.index(tuvalue),
            self.y_pos[i]:self.y_pos[i] + self.height[i],
            self.x_pos[i] : self.x_pos[i] + self.width[i]] = 1
        return tumap

    def setTuBoundaryFromBuffer(self, buffer):
        if len(buffer.shape)>1:
            for i in range(len(buffer.shape)):
                self.setTuBoundaryFromBuffer(buffer[:,:,i])
        for i in range(len(self.tulist[0])):
            buffer[self.y_pos[i]:self.y_pos[i] + self.height[i],
                   self.x_pos[i]] = 0
            buffer[self.y_pos[i],
                   self.x_pos[i] + self.x_pos + self.width] = 0
        return

    def __iadd__(self, other):
        self.tulist = np.concatenate((self.tulist,other.tulist), axis = 0)
        self.resetMember()
        return self

    def __add__(self, other):
        tulist = np.concatenate((self.tulist,other.tulist), axis = 0)
        return TuList(tulist)


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __eq__(self, other):
        if self.width==other.width and self.height==other.height:
            return True
        return False

    def getArea(self):
        return self.width*self.height

    def getCArea(self):
        return self.getCSize().getArea()

    def getCSize(self):
        return Size(self.width>>1, self.height>>1)

    def getBuf(self, dtype = 'float32'):
        return np.zeros((self.height, self.width),dtype=dtype)

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def repositionTo(self, newpos):
        self.x = newpos.x
        self.y = newpos.y

    def relativeTo(self, origin):
        self.x -= origin.x
        self.y -= origin.y


    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Position(x,y)

    def __eq__(self, other):
        if self.x==other.x and self.y==other.y:
            return True
        return False

class Area(Size, Position):
    def __init__(self, w, h, x, y):
        Size.__init__(self, w,h)
        Position.__init__(self, x,y)

    def __eq__(self, other):
        if Size.__eq__(self, other) and Position.__eq__(self, other):
            return True
        return False

    def getCArea(self):
        return Area(self.width>>1, self.height>>1,self.x, self.y)

    def getSliceArea(self):
        return [self.y, self.y + self.height, self.x, self.x + self.width]


class UniBuf:
    def __init__(self, buf, tulist = None):
        # Size.__init__(self, w, h)
        self.buf = buf
        self.tulist = tulist

    def FillArea(self, area, value):
        self.buf[area.getSliceArea()] = value

    def FillbyTulist(self, idx):
        for tu in self.tulist.T:
            self.FillArea(Area(*tu[:4]), tu[idx-4])

    def copyBufFromArea(self, area):
        return self.buf[area.getSliceArea()]





class UnitBuf:
    def __init__(self, ChromaFormatID, area,
                 OrigY = None, OrigCb = None, OrigCr = None,
                 PredY = None, PredCb = None, PredCr = None,
                 ReconY = None, ReconCb = None, ReconCr = None,
                 UnfilteredY = None, UnfilteredCb = None, UnfilteredCr = None,
                 tuList = None
                 ):
        self.area = area
        self.chromaFormat = ChromaFormatID
        self.carea = self.getCArea()
        self.original = [OrigY, OrigCb, OrigCr]
        self.prediction = [PredY, PredCb, PredCr]
        self.reconstruction = [ReconY, ReconCb, ReconCr]
        self.unfilteredRecon = [UnfilteredY, UnfilteredCb, UnfilteredCr]
        self.pelBuf = [self.original, self.prediction, self.reconstruction, self.unfilteredRecon]
        self.tulist = tuList

    def CopyBlock(self, others ,PictureFormatID):
        if PictureFormatID%3:
            self.pelBuf[PictureFormatID][:, others.carea.y:(others.carea.y+others.carea.h), others.carea.x:(others.carea.x + others.carea.w)] = others.pelBuf[PictureFormatID][:,:,:]
        else:
            self.pelBuf[PictureFormatID][:, others.area.y:(others.area.y+others.area.h), others.area.x:(others.area.x + others.area.w)] = others.pelBuf[PictureFormatID][:,:,:]
        return

    def CopyAll(self, others):
        for i in range(len(self.pelBuf)):
            if self.pelBuf[i] is not None:
                self.CopyBlock(others, i)


    def CopyArea(self, others, PictureFormatID, dstarea, srcarea):
        self.pelBuf[PictureFormatID][:, dstarea.y:(dstarea.y + dstarea.h), dstarea.x:(dstarea.x + dstarea.w)] = others[:,srcarea.y:(srcarea.y + srcarea.h), srcarea.x:(srcarea.x + srcarea.w)]
        return

    def getNewBuf(self):
        if self.chromaFormat == ChromaFormat.YCbCr4_0_0:
            return [self.area.getBuf(), None, None]
        elif self.chromaFormat == ChromaFormat.YCbCr4_2_0:
            csize = self.area.getCSize()
            return [self.area.getBuf(), csize.getBuf(), csize.getBuf()]
        elif self.chromaFormat == ChromaFormat.YCbCr4_4_4:
            return [self.area.getBuf(), self.area.getBuf(), self.area.getBuf()]
        else:
            assert 0

    def getCArea(self):
        if self.chromaFormat == ChromaFormat.YCbCr4_2_0:
            return self.area.getCArea()
        else:
            return self.area

    def setReshape1dTo2d(self, pic_kinds):
        if self.chromaFormat == ChromaFormat.YCbCr4_2_0:
            self.pelBuf[pic_kinds][0] = self.pelBuf[pic_kinds][0].reshape((self.area.height, self.area.width))
            self.pelBuf[pic_kinds][1] = myUtil.UpSamplingChroma(self.pelBuf[pic_kinds][1].reshape((self.carea.height, self.carea.width)))
            self.pelBuf[pic_kinds][2] = myUtil.UpSamplingChroma(self.pelBuf[pic_kinds][2].reshape((self.carea.height, self.carea.width)))

    def dropPadding(self, x, pad, isDeepCopy = False):
        if isDeepCopy:
            return copy.deepcopy(x[:,pad:-pad,pad:-pad])
        else:
            return x[:,pad:-pad,pad:-pad]

    def getPic(self, PictureFormatID):
        return self.pelBuf[PictureFormatID]



