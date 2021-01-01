using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SPHSolver: MonoBehaviour
{
    public int maxParticleNum;
    public int particleNum 
    { 
        get 
        {
            int cnt = 0;
            foreach(SPH_Rigidbody i in _obstacles)
            {
                cnt += i.particleNum;
            }
            return particles.Count + cnt; 
        } 
    }
    public float timeStep = 0.01f;
    public float kernelRadius = 1.0f;
    #region Precomputed Values
    public float kr2, kr3Inv, kr6Inv, kr9Inv;
    #endregion

    public float stiffness = 0.1f;
    public float restDensity = 1.0f;
    public Vector3 externalAcc = Vector3.zero;
    public float viscosity = 1.0f;
    public float tensionCoeff = 1.0f;
    public float surfaceThreshold = 0.001f;

    public Bounds gridBound;
    public Vector3Int gridSize = 32 * Vector3Int.one;

    ParticleComparer particleComparer = null;

    #region GPU Adaptaion
    public List<Particle> particles;

    public Particle[] arrayParticles;

    ComputeShader sphShader = null;
    private int kernelCellIdx;
    private int kernelFindNearby;
    private int kernelUpdateDensity;
    private int kernelUpdateVisosityForce;
    private int kernelAdvanceParticle;
    private int kernelInitParticle;
    public const int sphThreadGroupSize = 512;
    public int _sphthreadGroupNum;

    public ComputeBuffer _bufParticles;
    private ComputeBuffer _bufNeighborSpace;
    public ComputeBuffer _bufParticleNumPerCell;
    public ComputeBuffer _bufGridStartIdx;

    private int[] _neighborSpace;
    private int[] _particleNumPerCell;
    private int[] _gridStartIdx;
    private int _scanThreadGroupNum;

    public List<SPH_Rigidbody> _obstacles;
    private ComputeBuffer _bufObstacles;
    #endregion

    private void Start()
    {
        kr2 = kernelRadius * kernelRadius;
        kr3Inv = 1.0f / kr2 / kernelRadius;
        kr6Inv = kr3Inv * kr3Inv;
        kr9Inv = kr6Inv * kr3Inv;

        kernelCellIdx = sphShader.FindKernel("ComputeCellIdx");
        kernelFindNearby = sphShader.FindKernel("FindNearby");
        kernelUpdateDensity = sphShader.FindKernel("UpdateDensity");
        kernelUpdateVisosityForce = sphShader.FindKernel("UpdateForce");
        kernelAdvanceParticle = sphShader.FindKernel("AdvanceParticle");
        kernelInitParticle = sphShader.FindKernel("InitParticle");

        sphShader.SetFloat("timeStep", timeStep);
        sphShader.SetFloat("kernelRadius", kernelRadius);
        sphShader.SetInts("gridSize", gridSize.x, gridSize.y, gridSize.z);
        sphShader.SetFloat("kr1Inv", 1.0f / kernelRadius);
        sphShader.SetFloat("kr3Inv", kr3Inv);
        sphShader.SetFloat("kr6Inv", kr6Inv);
        sphShader.SetFloat("kr9Inv", kr9Inv);
        sphShader.SetFloat("stiffness", stiffness);
        sphShader.SetFloat("restDensity", restDensity);
        sphShader.SetFloats("externalAcc", externalAcc.x, externalAcc.y, externalAcc.z);
        sphShader.SetFloat("viscosity", viscosity);
        sphShader.SetFloat("tensionCoeff", tensionCoeff);
        sphShader.SetFloat("surfaceThreshold", surfaceThreshold);
        sphShader.SetFloat("eps", Mathf.Epsilon);
        sphShader.SetFloats("lowerBound", gridBound.min.x, gridBound.min.y, gridBound.min.z);
        sphShader.SetFloats("upperBound", gridBound.max.x, gridBound.max.y, gridBound.max.z);

        particleComparer = new ParticleComparer();

    }

    public void Submit2GPU()
    {
        _sphthreadGroupNum = Mathf.CeilToInt((float)particleNum / (float)sphThreadGroupSize);
        sphShader.SetInt("particleNum", particleNum);
        arrayParticles = particles.ToArray();
        _bufParticles = new ComputeBuffer(particleNum, Particle.stride);
        _bufParticles.SetData(arrayParticles);

        _bufNeighborSpace = new ComputeBuffer(particleNum * 3 * 3 * 3, 4);
        _neighborSpace = new int[particleNum * 3 * 3 * 3];
        _bufNeighborSpace.SetData(_neighborSpace);

        _bufParticleNumPerCell = new ComputeBuffer(gridSize.x * gridSize.y * gridSize.z, sizeof(int));
        _particleNumPerCell = new int[gridSize.x * gridSize.y * gridSize.z];
        _bufParticleNumPerCell.SetData(_particleNumPerCell);

        _bufGridStartIdx = new ComputeBuffer(gridSize.x * gridSize.y * gridSize.z, sizeof(int));
        _gridStartIdx = new int[gridSize.x * gridSize.y * gridSize.z];

        sphShader.SetBuffer(kernelCellIdx, "particles", _bufParticles);
        sphShader.SetBuffer(kernelCellIdx, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.Dispatch(kernelCellIdx, _sphthreadGroupNum, 1, 1);
        _bufParticles.GetData(arrayParticles);
        System.Array.Sort(arrayParticles, particleComparer);

        _bufParticleNumPerCell.GetData(_particleNumPerCell);
        _gridStartIdx[0] = _particleNumPerCell[0];
        for (int i = 1; i < _gridStartIdx.Length; ++i)
            _gridStartIdx[i] = _gridStartIdx[i - 1] + _particleNumPerCell[i];
        _bufGridStartIdx.SetData(_gridStartIdx);

        sphShader.SetBuffer(kernelFindNearby, "particles", _bufParticles);
        sphShader.SetBuffer(kernelFindNearby, "neighborSpace", _bufNeighborSpace);
        sphShader.SetBuffer(kernelFindNearby, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.SetBuffer(kernelFindNearby, "gridStartIdx", _bufGridStartIdx);
        sphShader.Dispatch(kernelFindNearby, _sphthreadGroupNum, 1, 1);

        sphShader.SetBuffer(kernelUpdateDensity, "particles", _bufParticles);
        sphShader.SetBuffer(kernelUpdateDensity, "neighborSpace", _bufNeighborSpace);
        sphShader.SetBuffer(kernelUpdateDensity, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.SetBuffer(kernelUpdateDensity, "gridStartIdx", _bufGridStartIdx);
        sphShader.Dispatch(kernelUpdateDensity, _sphthreadGroupNum, 1, 1);

        sphShader.SetBuffer(kernelUpdateVisosityForce, "particles", _bufParticles);
        sphShader.SetBuffer(kernelUpdateVisosityForce, "neighborSpace", _bufNeighborSpace);
        sphShader.SetBuffer(kernelUpdateVisosityForce, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.SetBuffer(kernelUpdateVisosityForce, "gridStartIdx", _bufGridStartIdx);
        sphShader.Dispatch(kernelUpdateVisosityForce, _sphthreadGroupNum, 1, 1);

        sphShader.SetBuffer(kernelInitParticle, "particles", _bufParticles);
        sphShader.SetBuffer(kernelInitParticle, "neighborSpace", _bufNeighborSpace);
        sphShader.SetBuffer(kernelInitParticle, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.SetBuffer(kernelInitParticle, "gridStartIdx", _bufGridStartIdx);
        sphShader.Dispatch(kernelInitParticle, _sphthreadGroupNum, 1, 1);
    }

    public bool AddParticle(float mass, Vector3 pos, Vector3 v)
    {
        if (particleNum >= maxParticleNum)
            return false;

        particles.Add(new Particle(mass, pos, v));

        return true;
    }

    public bool AddObstacle(SPH_Rigidbody obj)
    {
        if (particleNum >= maxParticleNum)
            return false;
        _obstacles.Add(obj);
        return true;
    }
    private void FixedUpdate()
    {

        sphShader.SetBuffer(kernelFindNearby, "particles", _bufParticles);
        sphShader.SetBuffer(kernelFindNearby, "neighborSpace", _bufNeighborSpace);
        sphShader.SetBuffer(kernelFindNearby, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.SetBuffer(kernelFindNearby, "gridStartIdx", _bufGridStartIdx);
        sphShader.Dispatch(kernelFindNearby, _sphthreadGroupNum, 1, 1);

        sphShader.SetBuffer(kernelUpdateDensity, "particles", _bufParticles);
        sphShader.SetBuffer(kernelUpdateDensity, "neighborSpace", _bufNeighborSpace);
        sphShader.SetBuffer(kernelUpdateDensity, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.SetBuffer(kernelUpdateDensity, "gridStartIdx", _bufGridStartIdx);
        sphShader.Dispatch(kernelUpdateDensity, _sphthreadGroupNum, 1, 1);

        sphShader.SetBuffer(kernelUpdateVisosityForce, "particles", _bufParticles);
        sphShader.SetBuffer(kernelUpdateVisosityForce, "neighborSpace", _bufNeighborSpace);
        sphShader.SetBuffer(kernelUpdateVisosityForce, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.SetBuffer(kernelUpdateVisosityForce, "gridStartIdx", _bufGridStartIdx);
        sphShader.Dispatch(kernelUpdateVisosityForce, _sphthreadGroupNum, 1, 1);

        sphShader.SetBuffer(kernelAdvanceParticle, "particles", _bufParticles);
        sphShader.SetBuffer(kernelAdvanceParticle, "neighborSpace", _bufNeighborSpace);
        sphShader.SetBuffer(kernelAdvanceParticle, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.SetBuffer(kernelAdvanceParticle, "gridStartIdx", _bufGridStartIdx);
        sphShader.Dispatch(kernelAdvanceParticle, _sphthreadGroupNum, 1, 1);

        sphShader.SetBuffer(kernelCellIdx, "particles", _bufParticles);
        sphShader.SetBuffer(kernelCellIdx, "particleNumPerCell", _bufParticleNumPerCell);
        sphShader.Dispatch(kernelCellIdx, _sphthreadGroupNum, 1, 1);
        _bufParticles.GetData(arrayParticles);
        System.Array.Sort(arrayParticles, particleComparer);
        _bufParticleNumPerCell.GetData(_particleNumPerCell);
        _gridStartIdx[0] = _particleNumPerCell[0];
        for (int i = 1; i < _gridStartIdx.Length; ++i)
            _gridStartIdx[i] = _gridStartIdx[i - 1] + _particleNumPerCell[i];
        _bufGridStartIdx.SetData(_gridStartIdx);
    }

    private void Free()
    {
        if(_bufNeighborSpace != null)
        {
            _bufNeighborSpace.Release();
            _bufNeighborSpace = null;
        }
        if (_bufParticles != null)
        {
            _bufParticles.Release();
            _bufParticles = null;
        }
        if (_bufParticleNumPerCell != null)
        {
            _bufParticleNumPerCell.Release();
            _bufParticleNumPerCell = null;
        }
        if (_bufGridStartIdx != null)
        {
            _bufGridStartIdx.Release();
            _bufGridStartIdx = null;
        }
    }
}
