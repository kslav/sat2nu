#!/usr/bin/env python

import torch
import torch.nn
import numpy as np



class UNet(torch.nn.Module):
    def __init__(self, dropout, kernel_size=3,shallow_net=False,use_deconv=True):
      super(UNet, self).__init__()
      #upload rsync
      # (use [(W−K+2P)/S]+1 to determine output shape)
      self.dropout = dropout
      self.shallow_net = shallow_net
      self.use_deconv = use_deconv
      self.kernel_size=kernel_size
      self.drop = torch.nn.Dropout(self.dropout)
      
      ###### Encoding Blocks 
      self.conv1 = torch.nn.Conv2d(1, 64, self.kernel_size,2,1) # 1 channel, add padding to ensure input is halved
      self.conv2 = torch.nn.Conv2d(64, 128, self.kernel_size,2,1) # increase channel size
      self.conv3 = torch.nn.Conv2d(128, 256, self.kernel_size,2,1)
      self.conv4 = torch.nn.Conv2d(256, 512, self.kernel_size,2,1) 
      self.conv5 = torch.nn.Conv2d(512, 512, self.kernel_size,2,1) #repeat two times
      self.conv6 = torch.nn.Conv2d(512, 512, self.kernel_size,2,0) 

      # used in Encoder only
      self.leakyrelu  = torch.nn.LeakyReLU(negative_slope=0.2)

      ##### Decoding Blocks
      self.deconv0 =  torch.nn.ConvTranspose2d(512, 512, 4,2) # for 7 layer U-net
      self.deconv0b = torch.nn.ConvTranspose2d(512, 512, 2,2) # for 8-layer Unet
      self.deconv1 = torch.nn.ConvTranspose2d(1024, 512, 2,2) #repeat three times
      # use batch4 here
      self.deconv2 = torch.nn.ConvTranspose2d(1024, 512, 2,2) 
      self.deconv3 = torch.nn.ConvTranspose2d(1024, 256, 2,2)
      self.deconv3b = torch.nn.ConvTranspose2d(512, 256, 2,2) 
      # use batch3 here 
      self.deconv4 = torch.nn.ConvTranspose2d(512, 128, 2,2) 
      # use batch2 here
      self.deconv5 = torch.nn.ConvTranspose2d(256, 64, 2,2) 
      self.deconv6 = torch.nn.ConvTranspose2d(128, 1, 2,2) 
      
      # used in the decoding block only
      self.relu  = torch.nn.ReLU()
      
      # Decoding block with upsampling+conv2d instead of deconvolutions:
      self.upsamp2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
      self.upsamp4 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
      self.conv0_decod = torch.nn.Conv2d(512,512,3,1,1)
      self.conv1_decod = torch.nn.Conv2d(1024,512,3,1,1)
      # use batch4 here
      self.conv2_decod = torch.nn.Conv2d(1024,512,3,1,1)
      self.conv3_decod = torch.nn.Conv2d(1024,256,3,1,1)
      self.conv3b_decod = torch.nn.Conv2d(512,256,3,1,1)
      # use batch3 here
      self.conv4_decod = torch.nn.Conv2d(512,128,3,1,1)
      # use batch2 here
      self.conv5_decod = torch.nn.Conv2d(256,64,3,1,1)
      self.conv6_decod = torch.nn.Conv2d(128,1,3,1,1)


    def forward(self, x): # replaces forward() in the __call__() functin of torch.nn.Module
      out = []
      if self.shallow_net==True:
        # here we'll test a shallower version of the network above
        #### Encoder
        #x input is size 256 x 256 x 1 
        x1 = self.leakyrelu(self.conv1(x)) #128x128x64
        # instantiate the other layers without drop out, then check for drop out
        x2 = self.leakyrelu(self.conv2(x1)) #64x64x128
        x3 = self.leakyrelu(self.conv3(x2)) #32x32x256
        x4 = self.conv4(x3) #16x16x512

        # If dropout is non-zero, apply dropout right after activation of each layer
        if self.dropout !=0:
          x1 = self.drop(self.leakyrelu(self.conv1(x))) #128x128x64
          x2 = self.drop(self.leakyrelu(self.conv2(x1))) #64x64x128
          x3 = self.drop(self.leakyrelu(self.conv3(x2))) #32x32x256
          x4 = self.conv4(x3) #16x16x512
          
        #### Decoder  
        # Remember, input channels at each layer double because of skip connections!
        # Not applying any drop out to decoding layers at this point
                      #input 16x16x512
        y4 = []
        y5 = []
        y6 = []
        y7 = []
        if self.use_deconv: # did not find difference in performance whether using deconvs or upsample+conv, so didn't implement for deeper u-net
          y4 = self.deconv3b(self.relu(x4)) #32x32x256
                                                            #input 32x32x512
          y5 = self.deconv4(self.relu(torch.cat((y4,x3),dim=1))) #64x64x128
                                                            #input 64x64x256
          y6 = self.deconv5(self.relu(torch.cat((y5,x2),dim=1))) #128x128x64
                                                            #input 128x128x128                       
          y7 =             self.deconv6(self.relu(torch.cat((y6,x1),dim=1))) #256x256x1 
        else:
          y4 = self.conv3b_decod(self.upsamp2(self.relu(x4))) #32x32x256
                                                            #input 32x32x512
          y5 = self.conv4_decod(self.upsamp2(self.relu(torch.cat((y4,x3),dim=1)))) #64x64x128
                                                            #input 64x64x256
          y6 = self.conv5_decod(self.upsamp2(self.relu(torch.cat((y5,x2),dim=1)))) #128x128x64
                                                            #input 128x128x128                       
          y7 =             self.conv6_decod(self.upsamp2(self.relu(torch.cat((y6,x1),dim=1)))) #256x256x1 
                                                                         

        out = y7 + x # with addition of input to output

      elif self.shallow_net == False:
        #### Encoder
        #x input is size 256 x 256 x 1 
        x1 = self.leakyrelu(self.batch1(self.conv1(x))) #128x128x64

        # instantiate the other layers without drop out, then check for drop out
        x2 = self.leakyrelu(self.batch2(self.conv2(x1))) #64x64x128
        x3 = self.leakyrelu(self.batch3(self.conv3(x2))) #32x32x256
        x4 = self.leakyrelu(self.batch4(self.conv4(x3))) #16x16x512
        x5 = self.leakyrelu(self.batch5(self.conv5(x4))) #8x8x512
        x6 = self.leakyrelu(self.batch5(self.conv5(x5))) #4x4x512
        x7 = self.conv6(x6)              #1x1x512 

        # If dropout is non-zero, apply dropout right after activation of each layer
        if self.dropout !=0:
          x1 = self.drop(self.leakyrelu(self.batch1(self.conv1(x)))) #128x128x64
          x2 = self.drop(self.leakyrelu(self.batch2(self.conv2(x1)))) #64x64x128
          x3 = self.drop(self.leakyrelu(self.batch3(self.conv3(x2)))) #32x32x256
          x4 = self.drop(self.leakyrelu(self.batch4(self.conv4(x3)))) #16x16x512
          x5 = self.drop(self.leakyrelu(self.batch5(self.conv5(x4)))) #8x8x512
          x6 = self.drop(self.leakyrelu(self.batch5(self.conv5(x5)))) #4x4x512
          x7 = self.conv6(x6)              #1x1x512 
          


        #### Decoder  
        # Remember, input channels at each layer double because of skip connections!
        # Not applying any drop out to decoding layers at this point

                           #input 1x1x512
        y1 = self.deconv0(self.relu(x7)) #4x4x512         #input 4x4x1024
        y2 = self.batch4(self.deconv1(self.relu(torch.cat((y1,x6),dim=1)))) #8x8x512
                                                          #input 8x8x1024
        y3 = self.batch4(self.deconv2(self.relu(torch.cat((y2,x5),dim=1)))) #16x16x512
                                                          #input 16x16x1024
        y4 = self.batch3(self.deconv3(self.relu(torch.cat((y3,x4),dim=1)))) #32x32x256
                                                          #input 32x32x512
        y5 = self.batch2(self.deconv4(self.relu(torch.cat((y4,x3),dim=1)))) #64x64x128
                                                          #input 64x64x256
        y6 = self.batch1(self.deconv5(self.relu(torch.cat((y5,x2),dim=1)))) #128x128x64
                                                          #input 128x128x128                       
        y7 =             self.deconv6(self.relu(torch.cat((y6,x1),dim=1))) #256x256x1 
                                                                           

        out = y7 + x # with addition of input to output
        #out = y7 # test case without input addition and see what happens...
         
      print("------> out.shape = ", out.shape)
      return out # return the output of the U-Net



