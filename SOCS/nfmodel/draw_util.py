
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

def color_map_gif(x,y,zlist,point_list):
    fig, ax = plt.subplots()
    sc = ax.scatter(point_list[0][0], point_list[0][1],color='r')
    z_min, z_max = -np.abs(zlist[0]).max(), np.abs(zlist[0]).max()
    pm = ax.pcolormesh(x, y, zlist[0], cmap ='Greens', vmin = 0, vmax = z_max)
    cbar = fig.colorbar(pm)
    total=len(point_list)
    def update(i):
        data=np.hstack((point_list[i][None,0], point_list[i][None,1]))
        sc.set_offsets(data)
        z=zlist[i]
        pm.set_array(z.ravel())
        sc.set_zorder(1)
        pm.set_zorder(0)

        # show_scatter(test_nocs_np[None,:],target_nocs[target_topk_index].detach().cpu().numpy())
    ani = animation.FuncAnimation(fig, update, frames=total)
    ani.save('./save3d/color_map.gif', writer='imagemagick', fps=2)
    plt.show()


def show_gif(pointsList1,pointsList2):
    from matplotlib import pyplot as plt
    import numpy as np
    import mpl_toolkits.mplot3d.axes3d as p3
    from matplotlib import animation
    fig = plt.figure()

    ax = p3.Axes3D(fig)
    ax.set_box_aspect([1,1,1])
    ax.set(xlim3d=(-300, 300), xlabel='X')
    ax.set(ylim3d=(-300, 300), ylabel='Y')
    ax.set(zlim3d=(-300, 300), zlabel='Z')
    azim=0
    if azim is not None:
        ax.azim = azim
    # if dist is not None:
    #     ax.dist = dist
    # if elev is not None:
    #     ax.elev = elev

    points1, = ax.plot(pointsList1[0][:,0], pointsList1[0][:,1], pointsList1[0][:,2], '.r')
    points2, = ax.plot(pointsList2[0][:,0], pointsList2[0][:,1], pointsList2[0][:,2], '.b')
    def update_points(i):
        # update properties
        points1.set_data(pointsList1[i][:,0], pointsList1[i][:,1])
        points1.set_3d_properties(pointsList1[i][:,2], 'z')

        points2.set_data(pointsList2[i][:,0], pointsList2[i][:,1])
        points2.set_3d_properties(pointsList2[i][:,2], 'z')

        # return modified artists
        return points1,points2
    ani=animation.FuncAnimation(fig, update_points, frames=50)
    ani.save('./interpo_gru/save3d/animation.gif', writer='imagemagick', fps=2)
    plt.show()

    return