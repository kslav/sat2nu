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
        if self.use_deconv:
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



## FORWARD OPERATOR FOR 8-LAYER U-Net

    # def forward(self, x): # replaces forward() in the __call__() functin of torch.nn.Module
        
    #     #### Encoder
    #     #x input is size 256 x 256 x 1 
    #     x1 = self.leakyrelu(self.batch1(self.conv1(x))) #128x128x64
        
    #     # instantiate the other layers, then check for dropout
    #     x2 = self.drop(self.leakyrelu(self.batch2(self.conv2(x1)))) #64x64x128
    #     x3 = self.drop(self.leakyrelu(self.batch3(self.conv3(x2)))) #32x32x256
    #     x4 = self.drop(self.leakyrelu(self.batch4(self.conv4(x3)))) #16x16x512
    #     x5 = self.drop(self.leakyrelu(self.batch5(self.conv5(x4)))) #8x8x512
    #     x6 = self.drop(self.leakyrelu(self.batch5(self.conv5b(x5)))) #4x4x512
    #     x7 = self.drop(self.leakyrelu(self.batch5(self.conv5c(x6)))) #2x2x512 
    #     x8 = self.conv5c(x7)                                         #1x1x512 

    #     # If dropout is non-zero, apply dropout right after activation of each layer
    #     if self.dropout !=0:
    #       x2 = self.leakyrelu(self.batch2(self.conv2(x1))) #64x64x128
    #       x3 = self.leakyrelu(self.batch3(self.conv3(x2))) #32x32x256
    #       x4 = self.leakyrelu(self.batch4(self.conv4(x3))) #16x16x512
    #       x5 = self.leakyrelu(self.batch5(self.conv5(x4))) #8x8x512
    #       x6 = self.leakyrelu(self.batch5(self.conv5b(x5))) #4x4x512
    #       x7 = self.leakyrelu(self.batch5(self.conv5c(x6))) #2x2x512
    #       x8 = self.conv5c(x7)                              #1x1x512 
        

    #     #### Decoder  
    #     # Remember, input channels at each layer double because of skip connections!
    #     # Not applying any drop out to decoding layers at this point

    #                        #input 1x1x512
    #     y1 = self.deconv0b(self.relu(x8)) #2x2x512         #input 2x2x1024
    #     y2 = self.batch4(self.deconv1(self.relu(torch.cat((y1,x7),dim=1)))) #4x4x512
    #                                                       #input 8x8x1024
    #     y3 = self.batch4(self.deconv2(self.relu(torch.cat((y2,x6),dim=1)))) #8x8x512
    #                                                       #input 8x8x1024
    #     y4 = self.batch4(self.deconv2(self.relu(torch.cat((y3,x5),dim=1)))) #16x16x512 ****
    #                                                       #input 16x16x1024
    #     y5 = self.batch3(self.deconv3(self.relu(torch.cat((y4,x4),dim=1)))) #32x32x256
    #                                                       #input 32x32x512
    #     y6 = self.batch2(self.deconv4(self.relu(torch.cat((y5,x3),dim=1)))) #64x64x128
    #                                                       #input 64x64x256
    #     y7 = self.batch1(self.deconv5(self.relu(torch.cat((y6,x2),dim=1)))) #128x128x64
    #                                                       #input 128x128x128                       
    #     y8 =             self.deconv6(self.relu(torch.cat((y7,x1),dim=1))) #256x256x1 (actually 1 x 1 x 224 x 224)
                                                                         

    #     out = self.relu(y8) + x # with addition of input to output
    #     #out = y7 # test case without input addition and see what happens...
       
  
    #     return out # return the output of the U-Net

## ORIGINAL FORWARD OPERATOR FOR 7-LAYER U-NET ##
# def forward(self, x): # replaces forward() in the __call__() functin of torch.nn.Module
        
#         #### Encoder
#         #x input is size 256 x 256 x 1 
#         x1 = self.leakyrelu(self.batch1(self.conv1(x))) #128x128x64
        
#         # instantiate the other layers, then check for dropout
#         x2 = 0
#         x3 = 0
#         x4 = 0 
#         x5 = 0
#         x6 = 0 
#         x7 = 0 

#         # If dropout is non-zero, apply dropout right after activation of each layer
#         if self.dropout !=0:
#           x2 = self.drop(self.leakyrelu(self.batch2(self.conv2(x1)))) #64x64x128
#           x3 = self.drop(self.leakyrelu(self.batch3(self.conv3(x2)))) #32x32x256
#           x4 = self.drop(self.leakyrelu(self.batch4(self.conv4(x3)))) #16x16x512
#           x5 = self.drop(self.leakyrelu(self.batch5(self.conv5(x4)))) #8x8x512
#           x6 = self.drop(self.leakyrelu(self.batch5(self.conv5(x5)))) #4x4x512
#           x7 = self.conv6(x6)              #1x1x512 
#         else:
#           x2 = self.leakyrelu(self.batch2(self.conv2(x1))) #64x64x128
#           x3 = self.leakyrelu(self.batch3(self.conv3(x2))) #32x32x256
#           x4 = self.leakyrelu(self.batch4(self.conv4(x3))) #16x16x512
#           x5 = self.leakyrelu(self.batch5(self.conv5(x4))) #8x8x512
#           x6 = self.leakyrelu(self.batch5(self.conv5(x5))) #4x4x512
#           x7 = self.conv6(x6)              #1x1x512 
        

#         #### Decoder  
#         # Remember, input channels at each layer double because of skip connections!
#         # Not applying any drop out to decoding layers at this point

#                            #input 1x1x512
#         y1 = self.deconv0(self.relu(x7)) #4x4x512         #input 4x4x1024
#         y2 = self.batch4(self.deconv1(self.relu(torch.cat((y1,x6),dim=1)))) #8x8x512
#                                                           #input 8x8x1024
#         y3 = self.batch4(self.deconv2(self.relu(torch.cat((y2,x5),dim=1)))) #16x16x512
#                                                           #input 16x16x1024
#         y4 = self.batch3(self.deconv3(self.relu(torch.cat((y3,x4),dim=1)))) #32x32x256
#                                                           #input 32x32x512
#         y5 = self.batch2(self.deconv4(self.relu(torch.cat((y4,x3),dim=1)))) #64x64x128
#                                                           #input 64x64x256
#         y6 = self.batch1(self.deconv5(self.relu(torch.cat((y5,x2),dim=1)))) #128x128x64
#                                                           #input 128x128x128                       
#         y7 =             self.deconv6(self.relu(torch.cat((y6,x1),dim=1))) #256x256x1 (actually 1 x 1 x 224 x 224)
                                                                         

#         out = self.relu(y7) + x # with addition of input to output
#         #out = y7 # test case without input addition and see what happens...
       
  
#         return out # return the output of the U-Net


#### OLD STUFF ####
 # OLD FORMULATION OF ENCODER
    # x1 = self.batch1(self.conv1(x))
    # x2 = self.batch2(self.conv2(self.leakyrelu(x1))) #64x64x128
    # x3 = self.batch3(self.conv3(self.leakyrelu(x2))) #32x32x256
    # x4 = self.batch4(self.conv4(self.leakyrelu(x3))) #16x16x512
    # x5 = self.batch5(self.conv5(self.leakyrelu(x4))) #8x8x512
    # x6 = self.batch5(self.conv5(self.leakyrelu(x5))) #4x4x512
    # x7 = self.conv6(self.leakyrelu(x6))              #1x1x512 
 # FOR DEBUGGING INPUT 
 # Reshape x to be of size [batch, channel, X, Y]
        #ndims = len(x.shape)
        #permute_shape = list(range(ndims))
        #permute_shape.insert(1, permute_shape.pop(-1))
        #x = x.permute(permute_shape)
        #temp_shape = x.shape
        #x = x.reshape((x.shape[0], -1, x.shape[-2], x.shape[-1]))
        ## FOR DEBUGGING: print x shape after reshaping
        #print("------> x.shape = ",x.shape)
        #_x = x.detach().cpu().numpy()
  ## FOR DEBUGGING ENCODER: Print output shape of each layer ##
        # print("------>x1.shape = ", x1.shape)
        # print("------>x1 max = ", torch.max(x1))
        # print("------>x2.shape = ", x2.shape)
        # print("------>x2 max = ", torch.max(x2))
        # print("------>x3.shape = ", x3.shape)
        # print("------>x3 max = ", torch.max(x3))
        # print("------>x4.shape = ", x4.shape)
        # print("------>x4 max = ", torch.max(x4))
        # print("------>x5.shape = ", x5.shape)
        # print("------>x5 max = ", torch.max(x5))
        # print("------>x6.shape = ", x6.shape)
        # print("------>x6 max = ", torch.max(x6))
        # print("------>x7.shape = ", x7.shape)
        # print("------>x7 max = ", torch.max(x7))

 ## FOR DEBUGGING DECODER: Print output shape of each layer ##
        # print("------->y1.shape=",y1.shape)
        # print("------>y1 max = ", torch.max(y1))
        # print("------->y2.shape=",y2.shape)
        # print("------>y2 max = ", torch.max(y2))
        # print("------->y3.shape=",y3.shape)
        # print("------>y3 max = ", torch.max(y3))
        # print("------->y4.shape=",y4.shape)
        # print("------>y4 max = ", torch.max(y4))
        # print("------->y5.shape=",y5.shape)
        # print("------>y5 max = ", torch.max(y5))
        # print("------->y6.shape=",y6.shape)
        # print("------>y6 max = ", torch.max(y6))
        # print("------->y7.shape=",y7.shape)
        # print("------>y7 max = ", torch.max(y7))


