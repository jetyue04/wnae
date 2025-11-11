for noise in 0.1 0.4 0.8; do
  job_name="wnae-job-toy2d-noise${noise//./}-redo"
  output_dir="output/toy2d_noise${noise//./}_redo"

  cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: axol1tl
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: tuner
          image: python:3.10-slim
          imagePullPolicy: Always
          volumeMounts:
            - name: axovol
              mountPath: /axovol
          workingDir: /axovol/wnae_10_14
          command:
            - "/bin/bash"
            - "-c"
            - |
              set -e
              python -m pip install --no-cache-dir -r requirements.txt
              python -m pip install numpy==1.23.5 scipy==1.13.1
              python -u train_toy.py --override data.N=1 data.D=2 data.noise_std=${noise} training.n_epochs=500 data.output="${output_dir}" | tee train_${job_name}.log
          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: 2
            limits:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: 2
      volumes:
        - name: axovol
          persistentVolumeClaim:
            claimName: axovol
EOF
done
