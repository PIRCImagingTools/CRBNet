from nipy import load_image, save_image
import numpy as np
import getpass
from nipy.core.apy import Image
from nipype.interfaces import fsl

user=getpass.getuser()

template_stack = "./res/template_T2.nii.gz"
CRB_BRAIN = "./res/CRB_BRAIN_Template.nii.gz"
CRB_TEMPLATE = "./res/CRB_Template.nii.gz"

RCRB=18 #higher value
LCRB=17 #lower value

class CRB_PREP(object):

    def __init__(self, parent_dir, orig_brain, man_seg, PCA):
        self.parent_dir = parent_dir
        self.orig_brain = parent_dir+'T2_Bias_Corrected/'+orig_brain
        self.man_seg = parent_dir+man_seg
        self.PCA = PCA

    def reg_brain(self):

        if self.PCA >= 44:
            self.outmatrix_crb = self.parent_dir+'reg_to_CRB_template.mat'
            flt = fsl.FLIRT()
            flt.inputs.in_file =  self.brain
            flt.inputs.reference = CRB_BRAIN
            flt.inputs.out_matrix_file = self.outmatrix_crb
            flt.inputs.out_file = self.parent_dir + 'reg_to_CRB_template.nii.gz'
            flt.cmdline()
            flt.run()

            #apply reg to segmentation

            app = fsl.FLIRT()
            app.inputs.in_file = self.man_seg
            app.inputs.reference = self.get_template()
            app.inpus.apply_xfm = True
            app.inputs.in_matrix_file =  self.outmatrix_crb
            app.inputs.out_file = self.parent_dir + 'manual_seg_reg_to_CRB_template.nii.gz'
            app.cmdline()
            app.run()

        else:
            #REG TO BEST PCA MATCH FIRST

            self.outmatrix_pca = self.parent_dir+'reg_to_PCA_template.mat'
            flt_pca = fsl.FLIRT()
            flt_pca.inputs.in_file =  self.brain
            flt_pca.inputs.reference = self.get_template()
            flt_pca.inputs.out_matrix_file = self.outmatrix_orig
            flt_pca.inputs.out_file = self.parent_dir + 'reg_to_PCA_template.nii.gz'
            flt_pca.cmdline()
            flt_pca.run()

            #apply reg to segmentation

            app_pca = fsl.FLIRT()
            app_pca.inputs.in_file = self.man_seg
            app_pca.inputs.reference = self.get_template()
            app_pca.inpus.apply_xfm = True
            app_pca.inputs.in_matrix_file =  self.outmatrix_pca
            app_pca.inputs.out_file = self.parent_dir + 'manual_seg_reg_to_PCA_template.nii.gz'
            app_pca.cmdline()
            app_pca.run()

            #REG PCA REG TO OLDEST

            self.outmatrix_CRB = self.parent_dir+'reg_to_CRB_template.mat'
            flt_crb = fsl.FLIRT()
            flt_crb.inputs.in_file =  self.parent_dir + 'reg_to_PCA_template.nii.gz'
            flt_crb.inputs.reference =  CRB_BRAIN
            flt_crb.inputs.out_matrix_file = self.outmatrix_CRB
            flt_crb.inputs.out_file = self.parent_dir + 'reg_to_CRB_template.nii.gz'
            flt_crb.cmdline()
            flt_crb.run()

            #apply reg to segmentation

            app_crb = fsl.FLIRT()
            app_crb.inputs.in_file = self.parent_dir + 'manual_seg_reg_to_PCA_template.nii.gz'
            app_crb.inpus.apply_xfm = True
            app_crb.inputs.in_matrix_file =  self.outmatrix_pca
            app_crb.inputs.out_file = self.parent_dir + 'manual_seg_reg_to_CRB_template.nii.gz'
            app_crb.cmdline()
            app_crb.run()


    def get_index(self):
        if self.PCA >= 44:
            return 16
        elif self.PCA <= 28:
            return 0
        else:
            return self.PCA - 28

    def get_template(self):
        outfile_nii = self.parent_dir+'/template.nii.gz'
        fslroi = fsl.ExtractROI(in_file=template_stack,
                            roi_file=outfile_nii,
                            t_min=self.get_index(),
                            t_size=1)
        fslroi.run()


    def get_crb(self, std_seg_nii, output):
        brain = load_image(std_seg_nii)

    def add_to_stack(std_crb_nii, stack, class_vector):
        crb = load_image(std_crb_nii)


if __name__ == '__main__':

    parent_dir = '/home/rafa/Neonates/CHD_132/'
    orig_brain =  parent_dir + 'T2_Bias_Corrected/'
    man_seg = parent_dir + 'CHD_132_ManualSeg.nii'
    PCA = 36

    prep = CRB_PREP(parent_dir, orig_brain, man_seg, PCA)
    prep.reg_brain()


    template = get_template(template_nii, get_index(PCA))
    reg_brain(orig_brain, man_seg, template)
