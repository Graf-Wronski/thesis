# Nonchalant musings on repository improvement

## Top 1: General
### Naming conventions
Naming conventions should be enforced. One ontological entity should have one
name. E.g. "future" and "schedule" refer to the same thing.

### Scientific practice
Unit models such as batteries should refer to the paper they were taken from.
Otherwise, a validation of their workings is hardly possible. Likewise for
optimization algorithms.

## Top 2: Ontology
### Signals
It is questionable if one does want an extra class for any new information in
a signal. The informational content can also be validated in a single Signal
class. If however the current ontology should be kept then coordinators should
deal with those more smoothly. If a NoneSignal has to be transformed at 
receiver end to a fictive FirstOrderSignal than what is the benefit of having
a NoneSignal?

    if isinstance(signal, sig_types.NoneSignal):
        signal = sig_types.FirstOrderSignal(np.zeros(forecast.fc_len))
    elif isinstance(signal, sig_types.FirstOrderSignal):
        # Signal type is already correct.
        pass

### Forecasts
Forecasts do not deliver inflexible load and inflexible generation separately
but only their difference. I think these should be kept separated at this level
because otherwise the forecasts decides implicitly about self-sufficiency,
which is not the level to do that.

## Top 3: Pipeline
### Forecasting
At the moment pypsa pv input series are used to model pv input. However, both
pv capacity and solar irradiation are available. Using those to calculate pv
generation appears to be more consistent.

### Plotting
Plotting part of pipeline has hardly an information about input data, such as
transformer limit and thus can not illustrate relevant relations between input
and output data.

## Top 4: Minors
### Class names
Class names sometimes do not reflect the logic of class inheritance.

In GrECCo:
A ``PlainGridFeeCoordinator`` is a ``FirstOrderCoordinatorBase`` is a 
``CoordinatorInterface``.

In Reality:
A ``PlainGridFeeCoordinator`` is a ``FirstOrderCoordinator`` is a 
``Coordinator``.

Something being a Coordinator, a Base and an Interface can lead to ontological
confusion. Same goes e.g. for Signal'Type'.

