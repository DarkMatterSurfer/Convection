

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools
import os 
from docopt import docopt
from configparser import ConfigParser
import sys
from glob import glob
path = os.path.dirname(os.path.abspath(__file__))[:-12]

if len(sys.argv) < 2:
    print('please provide config file')
    raise
else:
    configfile = sys.argv[1]
config = ConfigParser()
config.read(str(configfile))
global lx
lx = config.getfloat('param','Lx')
lz = config.getfloat('param','Lz')
sig = config.getfloat('param','sig')
name = config.get('param', 'name')
Ra = config.getfloat('param','Ra')
def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks_r = ['buoyancy_r', 'vorticity_r']
    tasks_c = ['buoyancy_c', 'vorticity_c']
    titlelist=['Buoyancy','Vorticity']
    scale = 3
    dpi = 200
    title_func = lambda sim_time:'Ra= '+f"{(Ra):.1e}" +' t = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    # Layout
    nrows, ncols = 2,1
    image = plot_tools.Box(lx, lz)
    pad = plot_tools.Frame(0.3, 0, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        zpop = []
        xpop = None
        keylist = file['scales'].keys()
        for i in keylist:
            if 'z_hash' in i:
                zpop.append(file['scales'][i][()])
            elif 'x_hash' in i:
                xpop = file['scales'][i][()]
        if len(zpop)<2:
            print('Unavailable zpop')
            raise
        if xpop is None: 
            print('Unavailable xpop')
            raise
        if zpop[0][0] < zpop[1][0]:
            zglob = np.concatenate((zpop[0],zpop[1]))
        else:
            zglob = np.concatenate((zpop[1],zpop[0]))
        for index in range(start, start+count):
            fig, axs = plt.subplots(1, 2)
            # fig.colorbar(c, ax=ax)
            for n in range(len(tasks_r)):
                # tasks_r = ['buoyancy_r', 'vorticity_r']
                # tasks_c = ['buoyancy_c', 'vorticity_c']
                dta_r=file['tasks'][tasks_r[n]][()]
                dta_c=file['tasks'][tasks_c[n]][()]
                globtask = np.concatenate((dta_c,dta_r),axis = 2)
                # Build subfigure axes
                ax = axs[n]
                ax.set_aspect('equal')
                ax.set_adjustable('box', share=True)
                #sigma lines
                ax.axhline(0.5+sig,linestyle='--',color='k',lw=0.45)
                ax.axhline(0.5-sig,linestyle='--',color='k',lw=0.45)
                c = ax.pcolor(xpop,zglob,globtask[index, ...].T, cmap='RdBu') #buoyancy
                ax.set_title(titlelist[n],pad=50)
                fig.colorbar(c, ax=ax,location='top',orientation='horizontal')
                # Call 3D plotting helper, slicing in time
                # dset = file['tasks'][task_r]
                # plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task_r, even_scale=True, visible_axes=False)
                # dset = file['tasks'][task_c]
                # plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task_c, even_scale=True, visible_axes=False)
        
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.44, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
            plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    
    output_path = pathlib.Path(path+"/"+name+"/frames").absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(glob(path+"/"+name+"/snapshots/*.h5"), main, output=output_path)

