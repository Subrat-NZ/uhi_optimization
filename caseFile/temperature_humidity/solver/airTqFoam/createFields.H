Info<< "Reading field T\n" << endl;

volScalarField T
(
    IOobject
    (
        "T",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field q\n" << endl;

volScalarField q
(
    IOobject
    (
        "q",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field qFlux\n" << endl;

volScalarField qFlux
(
    IOobject
    (
        "qFlux",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);


Info<< "Reading field U\n" << endl;

volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);


Info<< "Reading field nut\n" << endl;

volScalarField nut
(
    IOobject
    (
        "nut",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);


Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

dimensionedScalar nu("nu", dimViscosity, transportProperties);
dimensionedScalar Pr("Pr", dimless, transportProperties);
dimensionedScalar Prt("Prt", dimless, transportProperties);
dimensionedScalar Sc("Sc", dimless, transportProperties);
dimensionedScalar Sct("Sct", dimless, transportProperties);
dimensionedScalar rho("rho", dimDensity, transportProperties);

// Face flux of the momentum run (read from 0/phi if present, else flux(U))
#include "createPhi.H"

// Effective diffusivities (the wall values of nut are those of the
// momentum run's wall functions)
volScalarField alphaEff("alphaEff", nu/Pr + nut/Prt);
volScalarField DEff("DEff", nu/Sc + nut/Sct);
