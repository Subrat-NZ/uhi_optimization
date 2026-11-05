/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  8
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    airTqFoam

Description
    Steady transport of air temperature T and specific humidity q in a FROZEN
    flow field (passive scalars).  The velocity, the face flux phi and the
    turbulent viscosity nut come from the converged momentum (RANS) run and are
    read, never solved:

        div(phi, T) - laplacian(alphaEff, T) = 0     alphaEff = nu/Pr + nut/Prt
        div(phi, q) - laplacian(DEff,     q) = 0     DEff     = nu/Sc + nut/Sct

    Boundary conditions come from the files 0/T, 0/q and 0/qFlux written by
    map_surface_temperature.py:
      * T  : fixed wall values (mapped surface temperatures)
      * q  : `mixed` on the walls (fraction of a face covered by a saturated
             water surface -> valueFraction/refValue) plus an evaporation /
             transpiration mass flux [kg/m2/s] read from qFlux.  The flux is
             imposed through the refGradient of the `mixed` condition,
                 refGradient = qFlux / (rho * DEff_wall)
             with DEff_wall including the wall-function nut of the momentum run.
    Buoyancy is not modelled (the flow is not affected by T or q).

    Saturation (no relative humidity above 100 %):
      * limitEvaporation (default yes): the imposed flux qFlux is the flux into AMBIENT air (q = qRef).  Over a
        face with a flux the condition is made a Robin condition
            flux = qFlux * (qs - q_cell) / (qs - qRef),     qs = qsat(T_wall)
        (`mixed`: refValue = qs, refGradient 0, valueFraction = qFlux/(rho DEff deltaCoeffs (qs - qRef)), at most 1),
        so the flux is unchanged in ambient air and goes to zero as the air next to the surface saturates.
      * limitSupersaturation (default yes): the air cannot hold more than RHmax*qsat(T).  Cells above it get an
        implicit condensation sink  condensationRate*(q - RHmax*qsat(T))  (the excess is removed as liquid water),
        so the steady solution still converges.  Mass removed is printed.
    Optional entries of constant/transportProperties: pRef [Pa] (101325), qRef [kg/kg] (initial q), RHmax (1),
    condensationRate [1/s] (100), limitEvaporation, limitSupersaturation (yes/no).
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "simpleControl.H"
#include "wallFvPatch.H"
#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

//- Saturation specific humidity [kg/kg] at temperature TK [K] and pressure p [Pa]
//  (same Buck formula as tools/humidity_sources.py)
inline scalar qSatOf(const scalar TK, const scalar p)
{
    const scalar tc = TK - 273.15;
    const scalar es = 611.21*Foam::exp((18.678 - tc/234.5)*(tc/(257.14 + tc)));
    return 0.622*es/max(p - 0.378*es, 1.0);
}

int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    // Saturation settings (all optional)
    const scalar pRef = transportProperties.lookupOrDefault<scalar>("pRef", 101325.0);
    const Switch limitEvaporation
    (
        transportProperties.lookupOrDefault<Switch>("limitEvaporation", true)
    );
    const Switch limitSupersaturation
    (
        transportProperties.lookupOrDefault<Switch>("limitSupersaturation", true)
    );
    const scalar RHmax = transportProperties.lookupOrDefault<scalar>("RHmax", 1.0);
    const scalar condRate =
        transportProperties.lookupOrDefault<scalar>("condensationRate", 100.0);

    // Ambient specific humidity: the value the fluxes were computed for
    scalar qRef = transportProperties.lookupOrDefault<scalar>("qRef", -1.0);
    if (qRef < 0)
    {
        qRef = gAverage(q.primitiveField());
        Info<< "qRef not in transportProperties: using the mean initial q = "
            << qRef << " kg/kg" << endl;
    }

    Info<< "pRef " << pRef << " Pa, qRef " << qRef << " kg/kg, limitEvaporation "
        << limitEvaporation << ", limitSupersaturation " << limitSupersaturation
        << " (RHmax " << RHmax << ", condensationRate " << condRate << " 1/s)\n"
        << endl;

    // Evaporation / transpiration flux -> boundary condition of q on the wall faces
    scalar totalSource = 0;   // vapour emitted into ambient air [kg/s]
    {
        volScalarField::Boundary& qBf = q.boundaryFieldRef();

        forAll(qBf, patchi)
        {
            if
            (
                isA<wallFvPatch>(mesh.boundary()[patchi])
             && isA<mixedFvPatchScalarField>(qBf[patchi])
            )
            {
                mixedFvPatchScalarField& qp =
                    refCast<mixedFvPatchScalarField>(qBf[patchi]);

                const scalarField flux(qFlux.boundaryField()[patchi]);
                const scalarField Dw(DEff.boundaryField()[patchi]);
                const scalarField Tw(T.boundaryField()[patchi]);
                const scalarField dC(mesh.boundary()[patchi].deltaCoeffs());
                const scalarField& magSf = mesh.boundary()[patchi].magSf();

                // fixed flux (as before) ...
                qp.refGrad() = flux/(rho.value()*Dw);

                // ... made a Robin condition: flux -> 0 as q -> qs(T_wall)
                label nRobin = 0, nCapped = 0;
                scalar minRatio = 1;
                if (limitEvaporation)
                {
                    scalarField& vf = qp.valueFraction();
                    scalarField& rv = qp.refValue();

                    forAll(flux, facei)
                    {
                        if (flux[facei] <= 0) continue;

                        const scalar qs = qSatOf(Tw[facei], pRef);
                        const scalar dq = qs - qRef;
                        if (dq < 1e-5) continue;   // surface not wetter than the air: keep the fixed flux

                        // v = 1 would be a flux as large as the wall-side diffusion can carry
                        const scalar v =
                            flux[facei]/(rho.value()*Dw[facei]*dC[facei]*dq);
                        const scalar vfi = min(v, 1.0);

                        vf[facei] = vfi;
                        rv[facei] = qs;
                        qp.refGrad()[facei] = 0;

                        nRobin++;
                        if (v > 1) nCapped++;
                        minRatio = min(minRatio, vfi/max(v, small));
                    }
                }

                totalSource += gSum(flux*magSf);

                Info<< "patch " << mesh.boundary()[patchi].name()
                    << ": vapour flux " << gMin(flux) << " .. " << gMax(flux)
                    << " kg/m2/s, refGradient " << gMin(qp.refGrad())
                    << " .. " << gMax(qp.refGrad());
                if (limitEvaporation)
                {
                    Info<< ", saturation-limited faces " << returnReduce(nRobin, sumOp<label>())
                        << " (capped by wall diffusion: "
                        << returnReduce(nCapped, sumOp<label>())
                        << ", lowest flux ratio at ambient "
                        << returnReduce(minRatio, minOp<scalar>()) << ")";
                }
                Info<< endl;
            }
        }
    }
    Info<< "total vapour source into ambient air " << totalSource
        << " kg/s" << endl;

    Info<< "\nCalculating air temperature and specific humidity\n" << endl;

    while (simple.loop(runTime))
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix TEqn
            (
                fvm::div(phi, T)
              - fvm::laplacian(alphaEff, T)
            );

            TEqn.relax();
            TEqn.solve();

            // condensation sink in cells above RHmax*qsat(T) (implicit, active set)
            volScalarField sinkRate
            (
                IOobject("sinkRate", runTime.timeName(), mesh),
                mesh,
                dimensionedScalar("0", dimless/dimTime, 0)
            );
            volScalarField sinkSource
            (
                IOobject("sinkSource", runTime.timeName(), mesh),
                mesh,
                dimensionedScalar("0", dimless/dimTime, 0)
            );

            if (limitSupersaturation)
            {
                forAll(q, celli)
                {
                    const scalar qLim = RHmax*qSatOf(T[celli], pRef);
                    if (q[celli] > qLim*(1.0 - 1e-3))
                    {
                        sinkRate[celli] = condRate;
                        sinkSource[celli] = condRate*qLim;
                    }
                }
            }

            fvScalarMatrix qEqn
            (
                fvm::div(phi, q)
              - fvm::laplacian(DEff, q)
              + fvm::Sp(sinkRate, q)
             ==
                sinkSource
            );

            qEqn.relax();
            qEqn.solve();
        }

        // diagnostics: relative humidity of the air and condensed mass
        scalar RHmaxNow = 0;
        label nSat = 0;
        scalar condensed = 0;   // kg/s removed as liquid
        forAll(q, celli)
        {
            const scalar qs = qSatOf(T[celli], pRef);
            RHmaxNow = max(RHmaxNow, q[celli]/qs);
            if (q[celli] > RHmax*qs) nSat++;
            if (limitSupersaturation)
            {
                const scalar qLim = RHmax*qs;
                condensed +=
                    rho.value()*mesh.V()[celli]*condRate*max(q[celli] - qLim, 0.0);
            }
        }
        reduce(RHmaxNow, maxOp<scalar>());
        reduce(nSat, sumOp<label>());
        reduce(condensed, sumOp<scalar>());

        Info<< "T min/max = " << gMin(T) << " / " << gMax(T) << " K,  "
            << "q min/max = " << gMin(q) << " / " << gMax(q) << " kg/kg,  "
            << "RH max = " << 100*RHmaxNow << " %, cells above RHmax " << nSat
            << ", condensed " << condensed << " of " << totalSource << " kg/s"
            << nl << endl;

        runTime.write();
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
