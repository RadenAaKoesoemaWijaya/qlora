import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { WebSocketManager } from '@/lib/websocket';
import { formatDistanceToNow } from 'date-fns';
import { id } from 'date-fns/locale';

// Types for real-time monitoring data
type TrainingPhase = 'idle' | 'preparing' | 'training' | 'evaluating' | 'completed' | 'failed' | 'cancelled';

type TrainingMetrics = {
  step: number;
  epoch: number;
  loss: number;
  learning_rate: number;
  gradient_norm: number;
  gpu_memory_used_gb: number;
  gpu_memory_total_gb: number;
  cpu_percent: number;
  memory_percent: number;
  disk_usage_percent: number;
  timestamp: string;
};

type SystemMetrics = {
  cpu_percent: number;
  memory_percent: number;
  disk_usage_percent: number;
  gpu_count: number;
  gpus: Array<{
    index: number;
    name: string;
    memory_used_gb: number;
    memory_total_gb: number;
    utilization_percent: number;
    temperature: number;
  }>;
  timestamp: string;
};

type TrainingJob = {
  id: string;
  name: string;
  status: TrainingPhase;
  progress: number;
  current_step: number;
  total_steps: number;
  current_epoch: number;
  total_epochs: number;
  model_name: string;
  dataset_name: string;
  start_time: string;
  estimated_completion_time?: string;
  error_message?: string;
  recent_metrics: TrainingMetrics[];
};

type ErrorLog = {
  id: string;
  job_id: string;
  timestamp: string;
  level: 'error' | 'warning' | 'info';
  category: string;
  message: string;
  component: string;
  operation: string;
  resolved: boolean;
};

type Alert = {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  timestamp: string;
  job_id?: string;
  acknowledged: boolean;
};

interface RealTimeMonitoringDashboardProps {
  jobId?: string;
  onTrainingComplete?: (job: TrainingJob) => void;
  onError?: (error: Error) => void;
}

const RealTimeMonitoringDashboard: React.FC<RealTimeMonitoringDashboardProps> = ({
  jobId,
  onTrainingComplete,
  onError
}) => {
  const [trainingJob, setTrainingJob] = useState<TrainingJob | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [errorLogs, setErrorLogs] = useState<ErrorLog[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [chartTimeRange, setChartTimeRange] = useState(300); // 5 minutes
  
  const wsManager = useRef<WebSocketManager | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  
  // Initialize WebSocket connection
  useEffect(() => {
    const initWebSocket = () => {
      wsManager.current = new WebSocketManager();
      
      wsManager.current.on('connect', () => {
        setIsConnected(true);
        console.log('Connected to monitoring WebSocket');
        
        // Subscribe to job updates if jobId is provided
        if (jobId) {
          wsManager.current?.subscribe(`job:${jobId}`);
        }
        
        // Subscribe to system metrics
        wsManager.current?.subscribe('system:metrics');
        
        // Subscribe to error logs
        wsManager.current?.subscribe('logs:errors');
        
        // Subscribe to alerts
        wsManager.current?.subscribe('alerts:all');
      });
      
      wsManager.current.on('disconnect', () => {
        setIsConnected(false);
        console.log('Disconnected from monitoring WebSocket');
        
        // Attempt reconnection after 5 seconds
        if (reconnectTimeout.current) {
          clearTimeout(reconnectTimeout.current);
        }
        reconnectTimeout.current = setTimeout(initWebSocket, 5000);
      });
      
      wsManager.current.on('message', handleWebSocketMessage);
      
      wsManager.current.on('error', (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      });
      
      wsManager.current.connect();
    };
    
    initWebSocket();
    
    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      wsManager.current?.disconnect();
    };
  }, [jobId, onError]);
  
  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((message: any) => {
    try {
      const data = JSON.parse(message.data);
      
      switch (data.type) {
        case 'job_update':
          handleJobUpdate(data.payload);
          break;
        case 'system_metrics':
          handleSystemMetrics(data.payload);
          break;
        case 'error_log':
          handleErrorLog(data.payload);
          break;
        case 'alert':
          handleAlert(data.payload);
          break;
        default:
          console.warn('Unknown message type:', data.type);
      }
    } catch (error) {
      console.error('Error processing WebSocket message:', error);
    }
  }, []);
  
  // Handle job update
  const handleJobUpdate = useCallback((jobData: TrainingJob) => {
    setTrainingJob(jobData);
    
    // Check if training is complete
    if (jobData.status === 'completed' && onTrainingComplete) {
      onTrainingComplete(jobData);
    }
    
    // Update chart data
    if (jobData.recent_metrics && jobData.recent_metrics.length > 0) {
      // Chart data will be updated automatically via state
    }
  }, [onTrainingComplete]);
  
  // Handle system metrics update
  const handleSystemMetrics = useCallback((metrics: SystemMetrics) => {
    setSystemMetrics(metrics);
  }, []);
  
  // Handle error log
  const handleErrorLog = useCallback((errorLog: ErrorLog) => {
    setErrorLogs(prev => {
      const updated = [errorLog, ...prev];
      // Keep only last 100 error logs
      return updated.slice(0, 100);
    });
  }, []);
  
  // Handle alert
  const handleAlert = useCallback((alert: Alert) => {
    setAlerts(prev => {
      const updated = [alert, ...prev];
      // Keep only last 50 alerts
      return updated.slice(0, 50);
    });
  }, []);
  
  // Acknowledge alert
  const acknowledgeAlert = useCallback((alertId: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      )
    );
    
    // Send acknowledgment to backend
    wsManager.current?.send({
      type: 'acknowledge_alert',
      payload: { alert_id: alertId }
    });
  }, []);
  
  // Cancel training
  const cancelTraining = useCallback(() => {
    if (trainingJob && wsManager.current) {
      wsManager.current.send({
        type: 'cancel_training',
        payload: { job_id: trainingJob.id }
      });
    }
  }, [trainingJob]);
  
  // Get status color
  const getStatusColor = (status: TrainingPhase) => {
    switch (status) {
      case 'idle': return 'text-gray-500';
      case 'preparing': return 'text-blue-500';
      case 'training': return 'text-green-500';
      case 'evaluating': return 'text-purple-500';
      case 'completed': return 'text-green-600';
      case 'failed': return 'text-red-500';
      case 'cancelled': return 'text-orange-500';
      default: return 'text-gray-500';
    }
  };
  
  // Get status badge variant
  const getStatusBadgeVariant = (status: TrainingPhase) => {
    switch (status) {
      case 'idle': return 'secondary';
      case 'preparing': return 'default';
      case 'training': return 'default';
      case 'evaluating': return 'default';
      case 'completed': return 'success';
      case 'failed': return 'destructive';
      case 'cancelled': return 'outline';
      default: return 'secondary';
    }
  };
  
  // Get alert variant
  const getAlertVariant = (type: Alert['type']) => {
    switch (type) {
      case 'error': return 'destructive';
      case 'warning': return 'default';
      case 'info': return 'default';
      case 'success': return 'default';
      default: return 'default';
    }
  };
  
  // Filter metrics for time range
  const filterMetricsByTimeRange = (metrics: TrainingMetrics[]) => {
    const cutoff = new Date(Date.now() - chartTimeRange * 1000);
    return metrics.filter(m => new Date(m.timestamp) >= cutoff);
  };
  
  // Format time ago
  const formatTimeAgo = (timestamp: string) => {
    try {
      return formatDistanceToNow(new Date(timestamp), { addSuffix: true, locale: id });
    } catch {
      return 'Invalid time';
    }
  };
  
  // Render overview tab
  const renderOverview = () => (
    <div className="space-y-6">
      {/* Training Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Status Pelatihan</span>
            <Badge variant={trainingJob ? getStatusBadgeVariant(trainingJob.status) : 'secondary'}>
              {trainingJob?.status || 'Tidak ada pelatihan aktif'}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {trainingJob ? (
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Progres Keseluruhan</span>
                  <span>{trainingJob.progress.toFixed(1)}%</span>
                </div>
                <Progress value={trainingJob.progress} className="h-2" />
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-600">Langkah</p>
                  <p className="font-semibold">{trainingJob.current_step} / {trainingJob.total_steps}</p>
                </div>
                <div>
                  <p className="text-gray-600">Epoch</p>
                  <p className="font-semibold">{trainingJob.current_epoch} / {trainingJob.total_epochs}</p>
                </div>
              </div>
              
              <div className="text-sm space-y-1">
                <p><span className="text-gray-600">Model:</span> {trainingJob.model_name}</p>
                <p><span className="text-gray-600">Dataset:</span> {trainingJob.dataset_name}</p>
                <p><span className="text-gray-600">Dimulai:</span> {formatTimeAgo(trainingJob.start_time)}</p>
                {trainingJob.estimated_completion_time && (
                  <p><span className="text-gray-600">Estimasi selesai:</span> {formatTimeAgo(trainingJob.estimated_completion_time)}</p>
                )}
              </div>
              
              {trainingJob.error_message && (
                <Alert variant="destructive" className="mt-4">
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{trainingJob.error_message}</AlertDescription>
                </Alert>
              )}
              
              <div className="flex gap-2 mt-4">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={cancelTraining}
                  disabled={trainingJob.status === 'completed' || trainingJob.status === 'failed'}
                >
                  Batalkan Pelatihan
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => setAutoRefresh(!autoRefresh)}
                >
                  {autoRefresh ? 'Jeda Refresh' : 'Lanjutkan Refresh'}
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">Tidak ada pelatihan yang sedang berlangsung</p>
          )}
        </CardContent>
      </Card>
      
      {/* System Metrics Card */}
      <Card>
        <CardHeader>
          <CardTitle>Metrik Sistem</CardTitle>
        </CardHeader>
        <CardContent>
          {systemMetrics ? (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-600">{systemMetrics.cpu_percent.toFixed(1)}%</p>
                  <p className="text-sm text-gray-600">CPU</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-green-600">{systemMetrics.memory_percent.toFixed(1)}%</p>
                  <p className="text-sm text-gray-600">RAM</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-purple-600">{systemMetrics.disk_usage_percent.toFixed(1)}%</p>
                  <p className="text-sm text-gray-600">Disk</p>
                </div>
              </div>
              
              {systemMetrics.gpus.length > 0 && (
                <div className="space-y-2">
                  <h4 className="font-semibold">GPU Status</h4>
                  {systemMetrics.gpus.map((gpu) => (
                    <div key={gpu.index} className="flex items-center justify-between text-sm">
                      <span>{gpu.name}</span>
                      <div className="flex items-center gap-2">
                        <span>{gpu.memory_used_gb.toFixed(1)}GB / {gpu.memory_total_gb.toFixed(1)}GB</span>
                        <span className="text-xs text-gray-500">{gpu.utilization_percent}%</span>
                        <span className="text-xs text-gray-500">{gpu.temperature}°C</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">Tidak ada data metrik sistem</p>
          )}
        </CardContent>
      </Card>
      
      {/* Recent Alerts Card */}
      <Card>
        <CardHeader>
          <CardTitle>Alert Terbaru</CardTitle>
        </CardHeader>
        <CardContent>
          {alerts.filter(a => !a.acknowledged).length > 0 ? (
            <div className="space-y-2">
              {alerts.filter(a => !a.acknowledged).slice(0, 5).map((alert) => (
                <Alert key={alert.id} variant={getAlertVariant(alert.type)} className="mb-2">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <AlertTitle>{alert.title}</AlertTitle>
                      <AlertDescription>{alert.message}</AlertDescription>
                      <p className="text-xs text-gray-500 mt-1">
                        {formatTimeAgo(alert.timestamp)}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => acknowledgeAlert(alert.id)}
                      className="ml-2"
                    >
                      OK
                    </Button>
                  </div>
                </Alert>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">Tidak ada alert yang belum dikonfirmasi</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
  
  // Render metrics tab
  const renderMetrics = () => (
    <div className="space-y-6">
      {trainingJob && trainingJob.recent_metrics.length > 0 ? (
        <>
          {/* Time Range Selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Rentang Waktu:</span>
            {[60, 300, 600, 1800, 3600].map((seconds) => (
              <Button
                key={seconds}
                variant={chartTimeRange === seconds ? 'default' : 'outline'}
                size="sm"
                onClick={() => setChartTimeRange(seconds)}
              >
                {seconds >= 3600 ? `${seconds / 3600}j` : `${seconds / 60}m`}
              </Button>
            ))}
          </div>
          
          {/* Loss Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Loss selama Pelatihan</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={filterMetricsByTimeRange(trainingJob.recent_metrics)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString('id-ID')}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(timestamp) => new Date(timestamp).toLocaleString('id-ID')}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#2563eb" 
                    strokeWidth={2}
                    dot={false}
                    name="Loss"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="learning_rate" 
                    stroke="#16a34a" 
                    strokeWidth={2}
                    dot={false}
                    name="Learning Rate"
                    yAxisId="right"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          
          {/* GPU Memory Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Penggunaan Memori GPU</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={filterMetricsByTimeRange(trainingJob.recent_metrics)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString('id-ID')}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(timestamp) => new Date(timestamp).toLocaleString('id-ID')}
                    formatter={(value, name) => [
                      `${Number(value).toFixed(2)} GB`, 
                      name === 'gpu_memory_used_gb' ? 'Digunakan' : 'Total'
                    ]}
                  />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="gpu_memory_used_gb" 
                    stackId="1"
                    stroke="#2563eb" 
                    fill="#3b82f6"
                    name="Digunakan"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          
          {/* System Resource Usage */}
          <Card>
            <CardHeader>
              <CardTitle>Penggunaan Sumber Daya Sistem</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={filterMetricsByTimeRange(trainingJob.recent_metrics)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString('id-ID')}
                  />
                  <YAxis domain={[0, 100]} />
                  <Tooltip 
                    labelFormatter={(timestamp) => new Date(timestamp).toLocaleString('id-ID')}
                    formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Persentase']}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="cpu_percent" 
                    stroke="#dc2626" 
                    strokeWidth={2}
                    dot={false}
                    name="CPU %"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="memory_percent" 
                    stroke="#16a34a" 
                    strokeWidth={2}
                    dot={false}
                    name="RAM %"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="disk_usage_percent" 
                    stroke="#7c3aed" 
                    strokeWidth={2}
                    dot={false}
                    name="Disk %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </>
      ) : (
        <p className="text-gray-500 text-center py-8">Tidak ada data metrik untuk ditampilkan</p>
      )}
    </div>
  );
  
  // Render logs tab
  const renderLogs = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Log Error</h3>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setErrorLogs([])}
        >
          Bersihkan Log
        </Button>
      </div>
      
      {errorLogs.length > 0 ? (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {errorLogs.map((log) => (
            <Alert key={log.id} variant={log.level === 'error' ? 'destructive' : 'default'}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <AlertTitle>
                    [{log.level.toUpperCase()}] {log.category} - {log.component}
                  </AlertTitle>
                  <AlertDescription>
                    {log.message}
                    <div className="text-xs text-gray-500 mt-1">
                      <p>Operasi: {log.operation}</p>
                      <p>Waktu: {formatTimeAgo(log.timestamp)}</p>
                    </div>
                  </AlertDescription>
                </div>
                <Badge variant={log.resolved ? 'success' : 'outline'}>
                  {log.resolved ? 'Terselesaikan' : 'Belum terselesaikan'}
                </Badge>
              </div>
            </Alert>
          ))}
        </div>
      ) : (
        <p className="text-gray-500 text-center py-8">Tidak ada log error</p>
      )}
    </div>
  );
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Pemantauan Pelatihan Real-Time</h1>
          <p className="text-gray-600">Pantau pelatihan model dan metrik sistem secara real-time</p>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge variant={isConnected ? 'success' : 'destructive'}>
            {isConnected ? 'Terkoneksi' : 'Terputus'}
          </Badge>
          <div className="text-sm text-gray-500">
            {trainingJob ? `Job: ${trainingJob.name}` : 'Tidak ada pelatihan aktif'}
          </div>
        </div>
      </div>
      
      {/* Main Content */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Ikhtisar</TabsTrigger>
          <TabsTrigger value="metrics">Metrik</TabsTrigger>
          <TabsTrigger value="logs">Log</TabsTrigger>
          <TabsTrigger value="alerts">Alert</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-4">
          {renderOverview()}
        </TabsContent>
        
        <TabsContent value="metrics" className="space-y-4">
          {renderMetrics()}
        </TabsContent>
        
        <TabsContent value="logs" className="space-y-4">
          {renderLogs()}
        </TabsContent>
        
        <TabsContent value="alerts" className="space-y-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Semua Alert</h3>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setAlerts(prev => prev.map(a => ({ ...a, acknowledged: true })))}
              >
                Konfirmasi Semua
              </Button>
            </div>
            
            {alerts.length > 0 ? (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {alerts.map((alert) => (
                  <Alert key={alert.id} variant={getAlertVariant(alert.type)}>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <AlertTitle>
                          {alert.type === 'error' && '❌ '}
                          {alert.type === 'warning' && '⚠️ '}
                          {alert.type === 'info' && 'ℹ️ '}
                          {alert.type === 'success' && '✅ '}
                          {alert.title}
                        </AlertTitle>
                        <AlertDescription>{alert.message}</AlertDescription>
                        <p className="text-xs text-gray-500 mt-1">
                          {formatTimeAgo(alert.timestamp)}
                          {alert.job_id && ` • Job: ${alert.job_id}`}
                        </p>
                      </div>
                      {!alert.acknowledged && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => acknowledgeAlert(alert.id)}
                          className="ml-2"
                        >
                          OK
                        </Button>
                      )}
                    </div>
                  </Alert>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">Tidak ada alert</p>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RealTimeMonitoringDashboard;