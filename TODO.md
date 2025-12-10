# Gently TODO

## High Priority

- [ ] Fix numpy version issue
- [ ] Test Cellpose segmentation
- [ ] Fix CV agent errors
- [ ] Fix viz server UX issues

## Testing

- [ ] Test detection pipelines for embryo classification
- [ ] Test CV agent extensively

## Architecture / Investigation

- [ ] Rethink viz server layout
- [ ] Figure out datastore methods
- [ ] Figure out slow volume acquisition
  - Potentially rpyc netref slowing down volumes
  - Maybe combine start_server and simple_server?
  - Where is IO happening?
- [ ] Potentially upgrade to Micro-Manager 2.0
- [ ] Potentially introduce compaction of session - investigate context bloating
  - Check session 2e8c5aa9
- [ ] Investigate not being able to add detector tool
  - Check transcript of same session
- [ ] Investigate accuracy of timelapse time intervals
- [ ] Investigate whether we need to calibrate every embryo separately, or use the two-point from one embryo universally
- [ ] Distributed compute for CV agent - wire up two machines to schedule GPU tasks across systems
- [ ] ROI detection to reduce voxels passed through algorithms
- [ ] Ensure volume embryo images have good amount of padding
- [ ] Talk to Ryan about stage XY safe regions

## Viz Server Refactor

- [ ] Volume projection to 3D view with slider
- [ ] Potentially 4D for timelapse (2 sliders perpendicular to each other)
- [ ] Add indicator for timelapse tasks
- [ ] Add current stage position indicator with bottom camera view from initial image
- [ ] Add timeline visualization
- [ ] Add ability to zoom images
