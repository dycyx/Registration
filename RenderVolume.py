import glob
import os
import cv2
import numpy as np
import skimage.io
from mayavi import mlab
from skimage.measure import marching_cubes

VIDEODIR = ""
FRAMEDIR1 = "D:/Desktop/毕设/配准/image_reconstrution/rota_frame_png"
FRAMEDIR2 = "D:/Desktop/毕设/配准/image_reconstrution/recon_frame_png"


@mlab.animate(delay=10, ui=True)
def anim_rotate(sample_name):
    """
    旋转动画函数(可认为是帧)
    :param sample_name: 便于帧图片的命名
    """
    f = mlab.gcf()
    cam = f.scene.camera
    # 三种旋转方式（可详细调节）
    """
    azimuth:The azimuthal angle (in degrees, 0-360), i.e. the angle subtended by the position vector on a sphere projected on to the x-y plane with the x-axis.
    elevation：The zenith angle (in degrees, 0-180), i.e. the angle subtended by the position vector and the z-axis.
    distance:表示从焦点到放置相机的距离
    """
    # mlab.view(azimuth=None, elevation=0, distance=2000)
    mlab.view(azimuth=0, elevation=None, distance=2000)
    # mlab.view(azimuth=0, elevation=0, distance=2000)
    for i in range(10):
        # 将相机旋转 10 度,即相机每次拍摄时围绕拍摄物体所移动的角度
        cam.azimuth(10.0)
        # 在场景中调用渲染
        f.scene.render()
        print(mlab.view())
        filename = os.path.join(FRAMEDIR1, f"{int(os.path.basename(sample_name))}")
        filename = os.path.join(filename, f"{i}{'.png'}")
        # 保存帧
        mlab.savefig(filename=filename)  # size=(1024, 1024))
        yield


def anim_recon(sample_name, recon_nums):
    """
    重建动画函数(可认为是帧)
    :param recon_nums: 图片命名计数
    :param sample_name: 便于帧图片的命名
    """
    f = mlab.gcf()
    # 在场景中调用渲染
    f.scene.render()
    filename = os.path.join(FRAMEDIR2, f"{int(os.path.basename(sample_name))}")
    filename = os.path.join(filename, f"{recon_nums}{'.png'}")
    # 保存帧
    mlab.savefig(filename=filename)  # size=(1024, 1024))
    yield


def render_volume(vol, sample_name):
    """
    3D建模，并保存帧图
    @param vol:所有图片的列表
    """
    # vol = vol[30:-30, 30:-30, 10:-10]
    # print(vol.shape)
    # spacing=(.250, .250, 1.0)
    # x y z 间隔在这调整
    # TODO 有个参数Mask或许后续有用
    verts, faces, normals, values = marching_cubes(vol, 230, spacing=(1.0, 1.0, 10.0))
    # 设置mayavi窗口大小和背景颜色--黑色
    mlab.figure(size=(1024, 1200), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    # 3D建模
    tm = mlab.triangular_mesh([vert[0] for vert in verts],
                              [vert[1] for vert in verts],
                              [vert[2] for vert in verts],
                              faces,
                              )  # , colormap='hsv') colormap 可以改变可视化模型的样式
    # 自定义为23层颜色 原本256层
    tm.module_manager.scalar_lut_manager.lut.number_of_colors = vol.shape[2]
    # 自定义每层颜色
    lut = np.ones((vol.shape[2], 4), dtype=int)
    for i in range(vol.shape[2]):
        # RGBA
        lut[i] = [183, 108, 153, 255]
        # if i % 2 == 0:
        #     lut[i] = [183, 108, 153, 255]
        # else:
        #     lut[i] = [161, 123, 102, 255]
    tm.module_manager.scalar_lut_manager.lut.table = lut
    mlab.draw()
    # 记录帧图片
    anim_rotate(sample_name)
    # TODO 调用的图片慢慢增加，先不用实现
    # anim_recon(sample_name, 0)
    mlab.show()


def load_segmentation(sample_name):
    """
    加载配准后得到的--moved分割图
    @param sample_name:每个moved分割图所在的文件路径
    @return:每个moved分割图的像素强度矩阵
    """
    # 提取所有png格式的图片并以编号顺序排序
    frame_list = sorted(glob.glob(sample_name + '/*.png'), key=lambda x: int(x.split("_")[3]))

    # 排除的一帧
    # index = int((frame_list[0].split("/")[4]).split("_")[1])
    num_frame = len(frame_list)
    # first_frame = np.loadtxt(frame_list[0])
    first_frame = skimage.io.imread(frame_list[0])[:, :, 1:2]
    h, w, d = first_frame.shape
    block = np.zeros((h, w, num_frame), dtype=np.uint8)
    # np.savetxt("./output_occcf/same_size/0.txt", np.random.randint(0, 1, size=(512, 512)))
    for idx in range(num_frame):
        frame = skimage.io.imread(frame_list[idx])[:, :, 1:2].astype(np.uint8)
        block[:, :, idx] = frame[:, :, 0]
    print(block.shape)
    return block


def save_video(sample_name):
    """
    制作动画  利用获取3D图的图片帧
    @param sample_name:
    """
    # frame_list = sorted(glob.glob(sample_name + '/*.png'))
    # frame_list = sorted(glob.glob(sample_name + '/*.png'), key=lambda x: int(os.path.basename(x).split(".")[0]))
    frame_list = sorted(glob.glob(sample_name + '/*.png'), key=lambda x: int(os.path.basename(x).split(".")[0]))
    if len(frame_list) != 0:
        first_frame = cv2.imread(frame_list[0])
        # basebame 获取路径最后一个文件的名字， size参数必须是第一个大于第二个
        out = cv2.VideoWriter('segm_' + os.path.basename(sample_name) + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10,
                              (first_frame.shape[1], first_frame.shape[0]))
        for idx in range(len(frame_list)):
            # im = cv2.imread(os.path.join(sample_name, str(idx) + '_overlaid.png'))
            im = cv2.imread(frame_list[idx])
            out.write(im)
        out.release()
    else:
        return


def main_fc(video_flag=0):
    """
    主程序
    :param video_flag: 0 重建  1 生成旋转动画  2 生成重建动画
    """
    if video_flag == 0:
        sample_root_dir = r'D:\Desktop\毕设\配准\SimpleITK-Notebooks-master\Python\output_occcf\moved_images'
    elif video_flag == 1:
        sample_root_dir = r'D:\Desktop\毕设\配准\mage_reconstrution\rota_frame_png'
    elif video_flag == 2:
        sample_root_dir = r'D:\Desktop\毕设\配准\image_reconstrution\recon_frame_png'
    else:
        return
    # 获取当前文件夹下的所有文件路径
    sample_names = glob.glob(os.path.join(sample_root_dir, '*'))
    for sample_name in sample_names:
        print(sample_name)
        if video_flag == 0:
            # 加载分割
            block = load_segmentation(sample_name)
            # 建模，并保存帧图
            render_volume(block, sample_name)
        else:
            # 保存动画
            print("save")
            save_video(sample_name)


if __name__ == "__main__":
    # sample_root_dir = '/media/gjs/plum2t/Gjs_workstation/SimpleITK-Notebooks-master/Python/output_occcf/gray_imgs'
    # sample_root_dir = '/media/gjs/plum2t/Gjs_workstation/SimpleITK-Notebooks-master/Python/output_occcf/moved_images'
    # frame_root_dir = '/media/gjs/plum2t/Gjs_workstation/image_reconstrution/frame_png'

    # 0 建模，并保存帧图  1 生成旋转动画  2 生成建模动画（未实现）
    main_fc(0)

"""
# 保存分割好的图片
/media/gjs/plum2t/Gjs_workstation/SimpleITK-Notebooks-master/Python/output_occcf/use_mask_images
# 保存变换矩阵
/media/gjs/plum2t/Gjs_workstation/SimpleITK-Notebooks-master/Python/output_occcf/transform_metri
# 保存用于3D重建的图片
/media/gjs/plum2t/Gjs_workstation/SimpleITK-Notebooks-master/Python/output_occcf/moved_images
# 配准原始填充过的图片
/media/gjs/plum2t/Gjs_workstation/SimpleITK-Notebooks-master/Python/output_occcf/same_size
# 保存去除缩放参数后的G通道（fiexd_moved）合成图片
/media/gjs/plum2t/Gjs_workstation/SimpleITK-Notebooks-master/Python/output_occcf/moved_images
# 保存帧图片的位置
/media/gjs/plum2t/Gjs_workstation/image_reconstrution/frame_png

occcf 食道癌？ Oesophageal Cancer
TCGA 肝细胞癌
"""
