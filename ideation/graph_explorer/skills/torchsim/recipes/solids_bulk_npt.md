# Recipe: bulk crystalline solid at finite temperature

Build/import crystal -> static check -> relax positions/cell -> NVT -> optional NPT -> NVE/NVT production -> analyze lattice, energy, temperature, RDF, MSD, stress/elastic metrics -> export movie.

Starting values: timestep 0.001 ps; coordinate save stride 0.05-0.2 ps for visualization; production length depends on observable.
