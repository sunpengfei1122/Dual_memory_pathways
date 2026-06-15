ThisBuild / scalaVersion := "2.13.12"
ThisBuild / version := "0.1.0"
ThisBuild / organization := "com.dmpsnn"

lazy val root = (project in file("."))
  .settings(
    name := "dmp-snn-chisel",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % "5.3.0",
      "edu.berkeley.cs" %% "chiseltest" % "5.0.2" % "test"
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit"
    ),
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % "5.3.0" cross CrossVersion.full)
  )
