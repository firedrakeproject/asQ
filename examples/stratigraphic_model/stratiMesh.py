"""
Example of calling method:

mesh = stratiMesh(folder='output2', xdmfName = 'stratal_series', dispTime=5000.)
mesh.outputSteps(startTime=240000.,endTime=245000.)

"""

import os
import vtk
import numpy as np
import h5py


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class stratiMesh:
    """
    Class for creating irregular stratigraphic mesh from Carst outputs.
    """

    def __init__(self, folder=None, xdmfName = 'stratal_series', ncpus=1, layperstep=1, dispTime=None, verbose=False):
        """
        Initialization function which takes the folder path to Carst outputs
        and the number of CPUs used to run the simulation.
        Parameters
        ----------
        variable : folder
            Folder path to Carst outputs.
        variable : xdmfName
            Name of Carst stratigraphic grid outputs.
        variable: ncpus
            Number of CPUs used to run the simulation.
        variable: layperstep
            Number of layers created between each display
            interval (obtained from the XmL input file).
        variable: dispTime
            Time interval in years used to display Carst outputs.
        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.ncpus = ncpus

        self.x = None
        self.y = None
        self.elev = None
        self.dep = None
        self.th = None
        self.timestep = 0
        self.layperstep = 0
        self.laynb = 0
        self.startStep = None
        self.endStep = None
        self.rockNb = None

        # Assign file names
        self.h5TIN = 'tin.time'
        self.h5Strat = 'stratal.time'
        self.h5Strati = 'strati.time'
        self.xmffile = 'stratal.time'
        self.xdmfName = xdmfName+'.xdmf'
        self.dispTime = dispTime
        self.tnow = None

        return

    def _loadVTU(self):
        """
        Load VTU grid to extract cells connectivity and vertices position.
        Parameters
        ----------
        """

        pvtu = os.path.join(self.folder,'land_0.pvtu')
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(pvtu)
        reader.Update()
        output = reader.GetOutput()
        points = output.GetPoints()
        vtkData = points.GetData()
        cells = []
        nCells = output.GetNumberOfCells()
        for i in range(nCells):
            c = output.GetCell(i)
            p = c.GetPointIds()
            cells.append([p.GetId(j) for j in range(c.GetNumberOfEdges())])
        cells = np.array(cells)
        coords = np.array([vtkData.GetTuple3(i) for i in range(vtkData.GetNumberOfTuples())])

        # now get original land surface, which is held in the surface variable
        name = "starting_topo"
        try:
          pointdata=output.GetPointData()
          vtkdata=pointdata.GetScalars(name)
          vtkdata.GetNumberOfTuples()
        except:
          try:
            celldata=output.GetCellData()
            vtkdata=celldata.GetScalars(name)
            vtkdata.GetNumberOfTuples()
          except:
            raise Exception("ERROR: couldn't find point or cell scalar field data with name "+name+" in file "+pvtu+".")
        
        surface = np.array([vtkdata.GetTuple1(i) for i in range(vtkdata.GetNumberOfTuples())])
        coords[:,2] = surface

        # we therefore return the x,y,z plane and the cell connectivity

        return coords, cells

    def _loadStrati(self, step):
        """
        Load stratigraphic dataset.
        Parameters
        ----------
        variable : step
            Specific step at which the TIN variables will be read.
        variable: rank
            Stratigraphic grid for the considered CPU.
        """

        pvtu = 'output/S_'+str(step)+'.pvtu'
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(pvtu)
        reader.Update()
        output = reader.GetOutput()
        name = "depth"
        try:
          pointdata=output.GetPointData()
          vtkdata=pointdata.GetScalars(name)
          vtkdata.GetNumberOfTuples()
        except:
          try:
            celldata=output.GetCellData()
            vtkdata=celldata.GetScalars(name)
            vtkdata.GetNumberOfTuples()
          except:
            raise Exception("ERROR: couldn't find point or cell scalar field data with name "+name+" in file "+pvtu+".")
        
        palaeoH = np.array([vtkdata.GetTuple1(i) for i in range(vtkdata.GetNumberOfTuples())])
        name = "thickness"
        try:
          pointdata=output.GetPointData()
          vtkdata=pointdata.GetScalars(name)
          vtkdata.GetNumberOfTuples()
        except:
          try:
            celldata=output.GetCellData()
            vtkdata=celldata.GetScalars(name)
            vtkdata.GetNumberOfTuples()
          except:
            raise Exception("ERROR: couldn't find point or cell scalar field data with name "+name+" in file "+pvtu+".")
        
        thickness = np.array([vtkdata.GetTuple1(i) for i in range(vtkdata.GetNumberOfTuples())])

        # we now load in the current top surface
        pvtu = 'output/S_'+str(step)+'.pvtu'
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(pvtu)
        reader.Update()
        output = reader.GetOutput()
        name = "surface"
        try:
          pointdata=output.GetPointData()
          vtkdata=pointdata.GetScalars(name)
          vtkdata.GetNumberOfTuples()
        except:
          try:
            celldata=output.GetCellData()
            vtkdata=celldata.GetScalars(name)
            vtkdata.GetNumberOfTuples()
          except:
            raise Exception("ERROR: couldn't find point or cell scalar field data with name "+name+" in file "+pvtu+".")
        
        surface = np.array([vtkdata.GetTuple1(i) for i in range(vtkdata.GetNumberOfTuples())])

        return palaeoH, thickness, surface

    def _write_timestep(self, xt, yt, zt, cellt, layID, layH, layD,
                    clayID, clayD, clayH, step):
        """
        Write the HDF5 file containing the stratigraphic mesh variables.
        Parameters
        ----------
        variable : xt
            X-axis coordinates of the vertices.
        variable : yt
            Y-axis coordinates of the vertices.
        variable : zt
            Z-axis coordinates of the vertices.
        variable : cellt
            Wedge cells connectivity.
        variable : layID
            ID of each layer.
        variable : layH
            Thickness of each layer.
        variable : propR
            Proportion of each rock type in each layer.
        variable : layD
            Paleo depth informs about the elevation at time of deposition.
        variable : clayID
            Cell ID of each layer.
        variable : clayH
            Cell thickness of each layer.
        variable : cpropR
            Cell proportion of each rock type in each layer.
        variable : clayD
            Cell paleo depth informs about the elevation at time of deposition.
        variable: rank
            TIN file for the considered CPU.
        """

        h5file = self.folder+'/'+self.h5Strati+str(step)+'.hdf5'
        nElements = len(cellt[:,0])
        nNodes = len(xt)

        with h5py.File(h5file, "w") as f:

            # Write node coordinates and elevation
            f.create_dataset('coords',shape=(nNodes,3), dtype='float32', compression='gzip')
            f["coords"][:,0] = xt
            f["coords"][:,1] = yt
            f["coords"][:,2] = zt

            f.create_dataset('cells',shape=(nElements,6), dtype='int32', compression='gzip')
            f["cells"][:,:] = cellt

            f.create_dataset('layID',shape=(nNodes,1), dtype='int32', compression='gzip')
            f["layID"][:,0] = layID

            f.create_dataset('clayID',shape=(nElements,1), dtype='int32', compression='gzip')
            f["clayID"][:,0] = clayID

            f.create_dataset('layH',shape=(nNodes,1), dtype='float32', compression='gzip')
            f["layH"][:,0] = layH

            f.create_dataset('clayH',shape=(nElements,1), dtype='float32', compression='gzip')
            f["clayH"][:,0] = clayH

            f.create_dataset('layD',shape=(nNodes,1), dtype='float32', compression='gzip')
            f["layD"][:,0] = layD

            f.create_dataset('clayD',shape=(nElements,1), dtype='float32', compression='gzip')
            f["clayD"][:,0] = clayD


        xmf_file = self.folder+'/'+self.xmffile+str(step)+'.xmf'
        h5file = self.h5Strati+str(step)+'.hdf5'
        f= open(str(xmf_file),'w')

        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write(' <Domain>\n')

        pfile = self.h5Strati+str(step)+'.hdf5'
        f.write('      <Grid Name="Strat">\n')
        f.write('         <Time Type="Single" Value="%s"/>\n'%self.tnow)
        f.write('         <Topology Type="Wedge" NumberOfElements="%d" BaseOffset="1">\n'%nElements)
        f.write('          <DataItem Format="HDF" DataType="Int" ')
        f.write('Dimensions="%d 6">%s:/cells</DataItem>\n'%(nElements,h5file))
        f.write('         </Topology>\n')

        f.write('         <Geometry Type="XYZ">\n')
        f.write('          <DataItem Format="HDF" NumberType="Float" Precision="4" ')
        f.write('Dimensions="%d 3">%s:/coords</DataItem>\n'%(nNodes,h5file))
        f.write('         </Geometry>\n')

        f.write('         <Attribute Type="Scalar" Center="Node" Name="layer ID">\n')
        f.write('          <DataItem Format="HDF" NumberType="Int" ')
        f.write('Dimensions="%d 1">%s:/layID</DataItem>\n'%(nNodes,h5file))
        f.write('         </Attribute>\n')

        f.write('         <Attribute Type="Scalar" Center="Cell" Name="layer ID">\n')
        f.write('          <DataItem Format="HDF" NumberType="Int" ')
        f.write('Dimensions="%d 1">%s:/clayID</DataItem>\n'%(nElements,h5file))
        f.write('         </Attribute>\n')

        f.write('         <Attribute Type="Scalar" Center="Node" Name="paleo-depth">\n')
        f.write('          <DataItem Format="HDF" NumberType="Float" Precision="4" ')
        f.write('Dimensions="%d 1">%s:/layD</DataItem>\n'%(nNodes,h5file))
        f.write('         </Attribute>\n')

        f.write('         <Attribute Type="Scalar" Center="Cell" Name="paleo-depth">\n')
        f.write('          <DataItem Format="HDF" NumberType="float" ')
        f.write('Dimensions="%d 1">%s:/clayD</DataItem>\n'%(nElements,h5file))
        f.write('         </Attribute>\n')

        f.write('         <Attribute Type="Scalar" Center="Node" Name="layer th">\n')
        f.write('          <DataItem Format="HDF" NumberType="Float" Precision="4" ')
        f.write('Dimensions="%d 1">%s:/layH</DataItem>\n'%(nNodes,h5file))
        f.write('         </Attribute>\n')

        f.write('         <Attribute Type="Scalar" Center="Cell" Name="layer th">\n')
        f.write('          <DataItem Format="HDF" NumberType="float" ')
        f.write('Dimensions="%d 1">%s:/clayH</DataItem>\n'%(nElements,h5file))
        f.write('         </Attribute>\n')

        f.write('    </Grid>\n')
        f.write(' </Domain>\n')
        f.write('</Xdmf>\n')
        f.close()


        return

    def _write_xdmf(self):
        """
        Write the XDMF file which load and read the XMF parameters files for the requested steps.
        """

        f= open(self.folder+'/'+self.xdmfName,'w')

        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write(' <Domain>\n')
        f.write('    <Grid GridType="Collection" CollectionType="Temporal">\n')

        for p in range(self.startStep,self.endStep+1):
            xfile = self.xmffile+str(p)+'.xmf'
            f.write('      <xi:include href="%s" xpointer="xpointer(//Xdmf/Domain/Grid)"/>\n' %xfile)

        f.write('    </Grid>\n')
        f.write(' </Domain>\n')
        f.write('</Xdmf>\n')
        f.close()

        return

    def _writeAnteTopo(self, x, y, z, cells):
        """Write the antecedant topography to a file
        """
        newcells = cells+len(x)
        celltmp = np.concatenate((cells, newcells), axis=1)
        xtmp = np.concatenate((x[:,0], x[:,0]), axis=0)
        ytmp = np.concatenate((y[:,0], y[:,0]), axis=0)
        # create a surface just below the minimum value in z
        ztmp = np.concatenate((np.full((len(z[:,0])),np.amin(z)*1.1),z[:,0]), axis=0)

        # write the XMF
        xmf_file = self.folder+'/'+self.xmffile+'_antecedantTopography.xmf'
        f= open(str(xmf_file),'w')

        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write(' <Domain>\n')
        f.write('      <Grid Name="Mesh">\n')
        f.write('         <Topology Type="Wedge" NumberOfElements="%d" BaseOffset="1">\n'%len(xtmp))
        f.write('          <DataItem Format="HDF" DataType="Int" ')
        f.write('Dimensions="%d 6">%s_antecedantTopography.hdf5:/cells</DataItem>\n'%(len(celltmp[:,0]),self.h5Strati))
        f.write('         </Topology>\n')

        f.write('         <Geometry Type="XYZ">\n')
        f.write('          <DataItem Format="HDF" NumberType="Float" Precision="4" ')
        f.write('Dimensions="%d 3">%s_antecedantTopography.hdf5:/coords</DataItem>\n'%(len(xtmp),self.h5Strati))
        f.write('         </Geometry>\n')

        f.write('    </Grid>\n')
        f.write(' </Domain>\n')
        f.write('</Xdmf>\n')
        f.close()

        h5file = self.folder+'/'+self.h5Strati+'_antecedantTopography.hdf5'
        with h5py.File(h5file, "w") as f:

            # Write node coordinates and elevation
            f.create_dataset('coords',shape=(len(xtmp),3), dtype='float32', compression='gzip')
            f["coords"][:,0] = xtmp
            f["coords"][:,1] = ytmp
            f["coords"][:,2] = ztmp

            f.create_dataset('cells',shape=(len(celltmp[:,0]),6), dtype='int32', compression='gzip')
            f["cells"][:,:] = celltmp

        return




    def outputSteps(self, startTime=0, endTime=5000):
        """
        Define the steps that need to be visualise.
        Parameters
        ----------
        variable : startTime
            First Badlands output time to visualise.
        variable: endStep
            Last Badlands output time to visualise.
        """

        self.startTime = startTime
        self.endTime = endTime
        self.tnow = startTime

        self.startStep = int(startTime/self.dispTime)
        self.endStep = int(endTime/self.dispTime)
        if not os.path.isdir(self.folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        assert self.startStep<=self.endStep, 'ERROR: End step lower than Start step.'

        # Load initial PVTU grid for specific time step
        # Assumes no adaptivity
        coords, cells = self._loadVTU()
        x, y, z = np.hsplit(coords, 3)
        ptsNb = len(x)            
        
        # creat antecedant topography wedges
        self._writeAnteTopo(x, y, z, cells)

        for s in range(self.startStep, self.endStep+1):
            print('Processing layers at time [in years]: ', self.tnow, s)

            # this loads the sediment surface and the palaeo-
            # water depth (i.e. a rock property)
            rockProp = []
            thicknesses = []
            surfaces = []
            for ss in range(s, self.startStep-1, -1):
                paleoH, thickness, surface = self._loadStrati(ss)
                rockProp.append(paleoH)
                if len(thicknesses) > 1:
                    thicknesses.append(thickness)
                else:
                    thicknesses.append(thickness)
                surfaces.append(surface)


            nLayers = len(rockProp)
            rockProp = np.array(rockProp)
            thicknesses = np.array(thicknesses)
            surfaces = np.array(surfaces)

            # Layer number attribute
            layID = np.array([np.arange(nLayers+1,0,-1),]*ptsNb,dtype=int)
            ltmp = layID.flatten(order='F')

            # Thickness of each layer
            htmp = thicknesses.flatten(order='C')
            htmp = np.concatenate((np.zeros(ptsNb),htmp), axis=0)

            # Paleo-depth of each layer
            dtmp = rockProp.flatten(order='C')
            dtmp = np.concatenate((np.zeros(ptsNb),dtmp), axis=0)

            # Elevation of each layer
            # we have to limit previous steps thicknesses to that of the current top surface
            for tt in range(0,len(surfaces)):
                # build a mask
                mask = surfaces[tt]>surfaces[0] # our current top surface
                np.putmask(surfaces[tt], mask, surfaces[0])

            #cumH = thicknesses
            layZ = surfaces#np.add(cumH,z.reshape(1,cumH.shape[1]))
            ztmp = layZ.flatten(order='C')
            ztmp = np.concatenate((ztmp,z[:,0]), axis=0)

            # Creation of each layer coordinates
            xtmp = x[:, 0]
            ytmp = y[:, 0]
            ctmp = cells
            oldcells = ctmp

            # Cell layer index
            cellI = np.array([np.arange(nLayers+1, 1, -1)] * len(cells), dtype=int)
            cellItmp = cellI.flatten(order='F')

            for l in range(1, nLayers+1):
                xtmp = np.concatenate((xtmp, x[:, 0]), axis=0)
                ytmp = np.concatenate((ytmp, y[:, 0]), axis=0)

                # Creation of each layer elements
                newcells = oldcells+len(x)
                celltmp = np.concatenate((oldcells, newcells), axis=1)
                cellD = np.sum(dtmp[newcells-1], axis=1)/3.
                cellH = np.sum(htmp[newcells-1], axis=1)/3.
                oldcells = newcells
                if l == 1:
                    ctmp = np.copy(celltmp)
                    cellDtmp = np.copy(cellD)
                    cellHtmp = np.copy(cellH)
                else:
                    ctmp = np.concatenate((ctmp, celltmp), axis=0)
                    cellDtmp = np.concatenate((cellDtmp, cellD), axis=0)
                    cellHtmp = np.concatenate((cellHtmp, cellH), axis=0)

            # Create the HDF5 file
            self._write_timestep(xtmp, ytmp, ztmp,
                                 ctmp, ltmp, htmp, dtmp,
                                 cellItmp, cellDtmp, cellHtmp,
                                 s)
            self.tnow = self.tnow + self.dispTime

        # Create the XDMF file
        self._write_xdmf()

        return



mesh = stratiMesh('output', 'stratal_series', dispTime=4000.)
mesh.outputSteps(startTime=4000.,endTime=136000.)
