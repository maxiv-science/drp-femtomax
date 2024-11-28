# Viewers at FemtoMAX

## LiveViewer
This is meant to show you the **raw data** gathered from some detector, live obviously, at a rate of 1 Hz. \
Example: in a terminal at femtomax type `Liveviewer http://femtomax-zyla-12-daq.daq.maxiv.lu.se` 

The URL corresponds to the source of the raw data, in this case originating from a zyla-12 camera.
FemtoMAX have a detailed elogy entry about use and setup of the LiveViewer themselves, found [here](https://elogy.maxiv.lu.se/logbooks/165/entries/32899/?title=Liveviewer&authors=Felix).

## ModuleViewer
This is also a liveviewer, but of **processed data** rather than of raw data. The definition of processed data depends on the type of detector being considered:
- Balor: data as shown [here](https://gitlab.maxiv.lu.se/femtomax/drp-cmos/-/tree/main?ref_type=heads#single-photon-detection-on-balor) (Under "Single Photon Detection on Balor").
- Pilatus: data is the same as raw data, but can optionally be masked and / or acccumulated.
- Zyla: data as described [here](https://gitlab.maxiv.lu.se/femtomax/drp-cmos/-/tree/main?ref_type=heads#cluster-analysis-1) (under "Cluster Analysis").

 To run, type in a terminal at femtomax: `ModuleViewer -c DrpViewer http://femtomax-pipeline-reducer.daq.maxiv.lu.se`

 Again, the URL represents the source of the data, but note that it does not specifiy one detector.
 The ModuleViewer allows for the live viewing of processed data, regardless of which detector it came from. It shows whatever comes in from the end of the dranspose pipeline.

 The benefit is the possibility to accumulate images on top of one another (with the left invisible button - the right invisible button smears out single photon counts over a pixel neighborhood to make them more visible).

## HsdsViewer
This allows for viewing an updated version of the live h5-data from the source URL.
Some clarification: the source URL containing the data for live output behaves like a (web-)h5 file(*), always updating. The HsdsViewer allows for an updated / animated view of the data stored in the h5-file.

Example: in a terminal at femtomax type `HsdsViewer http://femtomax-pipeline-reducer.daq.maxiv.lu.se/ "/roi_means/ROI_3"` 

Shows you the updated / accumulated mean within the user-specified "`ROI_3`", that was created in the ModuleViewer GUI.
Note the addition of the extra "/" in the end of the URL.

----

(*): `silx` is the recommended tool used for viewing the data in an h5-file (or in a web-browser that we have convinced is an h5-file). 
If you don't have it installed, it is recommeneded you do so. Use a different terminal than the one from the previous step.

1. Make a dedicated conda environment, e.g., `conda create -n silx_test`, and activate it using `conda activate silx_test`
2. Install python 3.1x, with x=1 or x=2, no later, (`conda install python=3.12`), in this environment and git-clone from [the Hub](https://github.com/silx-kit/silx) directly.
3. In terminal, run `silx view http://foobar.maxiv.lu.se/` (Example added specifically to highlight the extra "/" at the end), where the URL denotes the source of the data you wish to view.

----

##  Restarting the pipeline:

If there is an error - one of the viewers is not working or updating - there are a few obvious steps to take.

1. Make sure the detector and its DCU are on and running. Each beamline should in principle have a protocol for starting, or restarting, the detector of interest.
2. Check the Tango device in jive: with Test Device, try the usual suspects:  State, Status, (Init).One of them should tell you where the error is coming from. The server can be restarted from  astor if necessary.
3. Ascertain you have a working LiveViewer. If you do not, the problem lies with the streaming-receiver. Its status can be checked in Kubernetes ([example](https://k8s.maxiv.lu.se/dashboard/c/c-m-gsgx4zjk/explorer/apps.deployment/femtomax-pilatus/femtomax-pilatus-daq#pods) for the Pilatus - the three dots next to the pod name allows you View Logs) and it can also be redeployed from there ([example](https://k8s.maxiv.lu.se/dashboard/c/c-m-gsgx4zjk/explorer/apps.deployment?q=femtomax-pilatus) for Pilatus).
4. If you have a working LiveViewer, but not a working ModuleViewer, the problem lies with dranspose. Similar to the streaming-receiver, its status can be checked in Kubernetes and it can also be redeployed from there ([example](https://k8s.maxiv.lu.se/dashboard/c/c-m-gsgx4zjk/explorer/apps.deployment?q=femtomax-pipeline) for Pilatus).
