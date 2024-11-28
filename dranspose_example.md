# Example: using dranspose for playing recaptured data
The goal of this section is to: 

- Run dranspose locally on your own machine.
- View some previously-captured data.
- Potentially alter the processing step that dranspose performs in feeding the processed data to the ModuleViewer.

Example is done on [femtomax/drp-cmos](https://gitlab.maxiv.lu.se/femtomax/drp-cmos).

## Install dranspose 

1. Make a dedicated conda environment, e.g., `conda create -n dranspose_test`, and activate it using `conda activate dranspose_test`
2. Install python (`conda install python`) and install dranspose using `pip install git+https://gitlab.maxiv.lu.se/scisw/daq-modules/dranspose.git`

## Install silx (optional)

`silx` is the recommended tool used for viewing the data in an h5-file (or in a web-browser that we have convinced is an h5-file). 
If you don't have it installed, it is recommeneded you do so. Use a different terminal than the one from the previous step.

1. Make a dedicated conda environment, e.g., `conda create -n silx_test`, and activate it using `conda activate silx_test`
2. Install python 3.1x, with x=1 or x=2, no later, (`conda install python=3.12`), in this environment and git-clone from [the Hub](https://github.com/silx-kit/silx) directly.
3. In terminal, run `silx view http://foobar.maxiv.lu.se/` (Example added specifically to highlight the extra "/" at the end), where the URL denotes the source of the data you wish to view. (Example: `http://femtomax-pipeline-reducer.daq.maxiv.lu.se/`)

## Clone Git repo and replay data

1. Git-clone the [drs-cmos repo](https://gitlab.maxiv.lu.se/femtomax/drp-cmos) locally.
2. There are some test data in the form of some Balor `.cbors` files present in the `data` folder (one `andor3 ingester` and one `sardana ingester`).
We are interested in [replaying captured data](https://dranspo.se/tutorials/analysis/#replaying-captured-data). The link reveals the relevant command to be: 
`LOG_LEVEL="DEBUG" dranspose replay -w "src.worker:CmosWorker" -r "src.reducer:CmosReducer" -f data/balor*.cbors --keep-alive --port=5000`

The variables `CmosReducer` and `CmosWorker` are class varible names found in the `reducer.py` and `worker.py` files in the `src` folder of the [drs-cmos repo](https://gitlab.maxiv.lu.se/femtomax/drp-cmos).
The port is explicitly given, because we forgot the default value and just set it to an educated guess instead of searching for it. (Update: turns out `5000` was the default value).

## Copy data (optional)

In our case, we wanted to play with some pilatus data, instead of the test balor data.
To do so, we copied the data at femtomax to the local data folder, which is done as follows (run from your local PC - added in case `scp` is new to you, or if the data folders are new to you):

`scp b-v-femtomax-cc-0:/data/staff/common/femtomax/dumps/pilatus* data/`

## Run

After the step "**Clone Git repos and replay data**", the terminal should state that it is awaiting a request (`INFO:dranspose.replay:waiting for event`).

In a new terminal, go to your silx conda environment (`conda activate silx_test`) and run `silx view http://localhost:5000`