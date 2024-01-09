## Development

The worker and reducer may be developed locally by replaying ingester recorded streams.

Provide the classes for the worker and reducer as well as the ingester files.
Optionally parameters may be provided in json or pickle format.

   dranspose replay -w "src.worker:FluorescenceWorker" -r "src.reducer:FluorescenceReducer" -f ../contrast_ingest.pkls ../xspress_ingest.pkls -p ../params.json